import torch
from torch import nn
import torch.nn.functional as F
import math
import timm
import config as CFG
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            
            # print(out_normal.shape, out_diff.shape)
            return out_normal - self.theta * out_diff

class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Diag_Cross, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:,:,:,0], tensor_zeros, self.conv.weight[:,:,:,1], tensor_zeros, self.conv.weight[:,:,:,2], tensor_zeros,\
                                  self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=1)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros, self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias,
                              stride=self.conv.stride, padding=1)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

def CDHV_ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, CD_Conv=Conv2d_cd):
    return nn.Sequential(
        CD_Conv(in_channels, out_channels, kernel_size,
                stride, padding, theta=0.7, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

def CDDC_ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, CD_Conv=Conv2d_cd):
    return nn.Sequential(
        CD_Conv(in_channels, out_channels, kernel_size,
                stride, padding, theta=0.3, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

def ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding, dilation, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

class GEM(nn.Module):
    def __init__(self, in_channels=5, out_channels=3):
        super(GEM, self).__init__()
        self.branch1 = CDHV_ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2)
        self.branch2 = CDDC_ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2)
        self.branch_last = ConvBNReLU(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        # print(out1.shape, out2.shape)
        out = torch.cat([out1, out2], dim=1)
        out = self.branch_last(out)
        # print(out.shape)
        return out

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x

class ImageEncoder_resnet50_ViT(nn.Module):
    """
    Encode images to a fixed size vector and pass through a Transformer block
    """
    def __init__(self, model_name='resnet50', pretrained=True, trainable=True, embed_dim=2048, num_heads=4, ff_dim=4096):
        super().__init__()
        self.conv = GEM(5, 3)
        self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        x = self.conv(x)
        x = self.model(x) # (B, 2048)
        x = x.unsqueeze(0)  # Add sequence dimension
        x = self.transformer_block(x)  # (B, 2048)
        x = x.squeeze(0)  # Remove sequence dimension
        return x

class PhenoProfiler_MSE(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        num_classes=CFG.num_classes,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet50_ViT()
        self.image_projection = ProjectionHead(embedding_dim=2048) #aka the input dim, 2048 for resnet50
        self.temperature = temperature
        self.classifier = nn.Linear(672, num_classes)

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        image_embeddings = self.image_projection(image_features)
        
        # print(batch["embedding"].shape)
        spot_embeddings = batch["embedding"]
        # print(spot_embeddings.shape)

        # Calculating the MSE Loss
        mse_loss = nn.MSELoss()
        mse_loss_value = mse_loss(image_embeddings, spot_embeddings)

        return mse_loss_value

class PhenoProfiler(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        num_classes=CFG.num_classes,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet50_ViT()
        self.image_projection = ProjectionHead(embedding_dim=2048) #aka the input dim, 2048 for resnet50
        self.temperature = temperature
        self.classifier = nn.Linear(672, num_classes)

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        image_embeddings = self.image_projection(image_features)
        
        # print(batch["embedding"].shape)
        spot_embeddings = batch["embedding"]
        # print(spot_embeddings.shape)

        ### calculate class loss      
        labels = batch["class"]
        logits = self.classifier(image_embeddings)
        class_loss = F.cross_entropy(logits, labels)

        # Calculating the contrastive Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        
        targets = F.softmax((images_similarity + spots_similarity) / 2 * self.temperature, dim=-1)    
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        
        contrastive_loss =  images_loss.mean()

        # Calculating the MSE Loss
        mse_loss = nn.MSELoss()
        mse_loss_value = mse_loss(image_embeddings, spot_embeddings)

        # Defining the weight coefficients
        alpha = 1
        beta = 100
        gamma = 0.1

        # Calculating the total loss 
        # print('loss:', contrastive_loss, mse_loss_value, class_loss)
        total_loss = alpha * contrastive_loss + beta * mse_loss_value + gamma * class_loss

        return total_loss
