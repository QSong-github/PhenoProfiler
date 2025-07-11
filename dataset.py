import os
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder

class PDD_Img_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, embedding_path, CSV_path):
        self.image_path = image_path
        self.embedding_path = embedding_path
        self.CSV_file = pd.read_csv(CSV_path)
        self.has_source = 'Source' in self.CSV_file.columns  # train single dataset = no  Source; train all datasets = Source

    def encode_labels(self):
        column_name = 'Treatment' if 'Treatment' in self.CSV_file.columns else 'pert_name' if 'pert_name' in self.CSV_file.columns else 'Compound'
        if column_name is None:
            raise ValueError("CSV file must contain 'Treatment', 'pert_name', or 'Compound' column")
        self.CSV_file['encoded_labels'] = self.label_encoder.fit_transform(self.CSV_file[column_name])

    def load_image(self, img_name, source=None):
        if source:
            # img_path = os.path.join(self.image_path, source, "original", 'images', img_name)
            img_path = os.path.join(self.image_path, source, 'images', img_name.replace('tif', 'png'))
        else:
            img_path = os.path.join(self.image_path, img_name.replace('tif', 'png'))
            # img_path = os.path.join(self.image_path, "original", 'images', img_name)
        return Image.open(img_path)

    def __getitem__(self, idx):
        item = {}
        channels = ['DNA', 'ER', 'RNA', 'AGP', 'Mito']
        if self.has_source:
            source = self.CSV_file.loc[idx, 'Source']
            images_list = [self.load_image(self.CSV_file.loc[idx, channel], source) for channel in channels]
        else:
            images_list = [self.load_image(self.CSV_file.loc[idx, channel]) for channel in channels]
        
        images = np.stack(images_list, axis=0)
        resized_image = resize(images, (5, 448, 448), anti_aliasing=True)
        preprocess = transforms.Compose([transforms.ToTensor()])
        resized_image_tensor = preprocess(resized_image.transpose(1, 2, 0))

        item['image'] = resized_image_tensor.float()
        
        return item

    def __len__(self):
        return self.CSV_file.shape[0]


class PDDDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, embedding_path, CSV_path):
        self.image_path = image_path
        self.embedding_path = embedding_path
        self.CSV_file = pd.read_csv(CSV_path)
        self.label_encoder = LabelEncoder()
        self.encode_labels()
        self.has_source = 'Source' in self.CSV_file.columns  # train single dataset = no  Source; train all datasets = Source

    def encode_labels(self):
        column_name = 'Treatment' if 'Treatment' in self.CSV_file.columns else 'pert_name' if 'pert_name' in self.CSV_file.columns else 'Compound'
        if column_name is None:
            raise ValueError("CSV file must contain 'Treatment', 'pert_name', or 'Compound' column")
        self.CSV_file['encoded_labels'] = self.label_encoder.fit_transform(self.CSV_file[column_name])

    def load_image(self, img_name, source=None):
        if source:
            # img_path = os.path.join(self.image_path, source, "original", 'images', img_name)
            img_path = os.path.join(self.image_path, source, 'images', img_name.replace('tif', 'png'))
        else:
            img_path = os.path.join(self.image_path, img_name.replace('tif', 'png'))
            # img_path = os.path.join(self.image_path, "original", 'images', img_name)
        return Image.open(img_path)

    def __getitem__(self, idx):
        item = {}
        channels = ['DNA', 'ER', 'RNA', 'AGP', 'Mito']
        if self.has_source:
            source = self.CSV_file.loc[idx, 'Source']
            images_list = [self.load_image(self.CSV_file.loc[idx, channel], source) for channel in channels]
        else:
            images_list = [self.load_image(self.CSV_file.loc[idx, channel]) for channel in channels]
        
        images = np.stack(images_list, axis=0)
        resized_image = resize(images, (5, 448, 448), anti_aliasing=True)
        preprocess = transforms.Compose([transforms.ToTensor()])
        resized_image_tensor = preprocess(resized_image.transpose(1, 2, 0))

        if self.has_source:
            embedding_path = os.path.join(
            self.embedding_path, 
            self.CSV_file.loc[idx, 'Source'].upper(), 
            str(self.CSV_file.loc[idx, 'Metadata_Plate']), 
            str(self.CSV_file.loc[idx, 'Metadata_Well']), 
            str(self.CSV_file.loc[idx, 'Metadata_Site']), 
            'PhenoProfiler_embeddings.npy'
            )
        else:
            embedding_path = os.path.join(
            self.embedding_path, 
            str(self.CSV_file.loc[idx, 'Metadata_Plate']), 
            str(self.CSV_file.loc[idx, 'Metadata_Well']), 
            str(self.CSV_file.loc[idx, 'Metadata_Site']), 
            'PhenoProfiler_embeddings.npy'
            )

        # print(embedding_path)
        if embedding_path.endswith('.npz'):
            with np.load(embedding_path) as data:
                features = data["features"]
                embedding = np.median(features[~np.isnan(features).any(axis=1)], axis=0)
        else:
            features = np.load(embedding_path)
            embedding = np.median(features[~np.isnan(features).any(axis=1)], axis=0)
        # print(embedding.shape)
        item['image'] = resized_image_tensor.float()
        item['embedding'] = torch.tensor(embedding).float()
        item['class'] = torch.tensor(self.CSV_file.loc[idx, 'encoded_labels']).long()

        return item

    def __len__(self):
        return self.CSV_file.shape[0]



class NoendDataset(torch.utils.data.Dataset):
    '''
    noendDataset(image_path = "/data/boom/cpg0019/broad/",
               embedding_path = "/data/boom/cpg0019/broad/workspace_dl/embeddings/105281_zenodo7114558/",
            #    CSV_path = "/data/boom/cpg0019/broad/workspace_dl/metadata/sc-metadata-fil.csv")
    '''
    def __init__(self, image_path, embedding_path, CSV_path):

        self.image_path = image_path
        self.embedding_path = embedding_path
        self.CSV_file = pd.read_csv(CSV_path)
        
        self.label_encoder = LabelEncoder()
        self.encode_labels()

    def encode_labels(self):
        if 'Treatment' in self.CSV_file.columns:
            column_name = 'Treatment'
        elif 'pert_name' in self.CSV_file.columns:
            column_name = 'pert_name'
        else:
            raise ValueError("CSV file must contain either 'Treatment' or 'pert_name' column")

        self.CSV_file['encoded_labels'] = self.label_encoder.fit_transform(self.CSV_file[column_name])
        
    def __getitem__(self, idx):
        item = {}
        # 获取图像位置，然后读取，然后按照坐标拆分，得到5张图象，然后叠加
        img_path = self.image_path + str(self.CSV_file.loc[idx, 'Image_Name'])[6:]
        All_img = Image.open(img_path)
        # print(img_path, All_img.size)

        # 分割大图像为6个子图像，每个子图像的尺寸为（160，160）
        sub_images = []
        for i in range(6):
            left = i * 160
            upper = 0
            right = left + 160
            lower = upper + 160
            sub_image = All_img.crop((left, upper, right, lower))
            sub_images.append(sub_image)
        
        # 按照通道叠加前面5张子图像在一起
        combined_image = np.stack(sub_images[:5], axis=0)
        # print(combined_image.shape)
        resized_image = resize(combined_image, (5, 224, 224), anti_aliasing=True)

        # embedding
        # /data/boom/cpg0019/broad/workspace_dl/embeddings/105281_zenodo7114558/BBBC022/20585/A01/1
        path = os.path.dirname(self.CSV_file.loc[idx, 'Image_Name'][22:])
        # embedding_path = os.path.join(self.embedding_path, str(path), 'embedding.npz')
        # print("embedding_path:", self.embedding_path, path, embedding_path)
        # with open(embedding_path, "rb") as data:
        #     info = np.load(data)
        #     cells = np.array(np.copy(info["features"]))
        #     embedding = cells[~np.isnan(cells).any(axis=1)]
        #     # embedding = np.median(embedding, axis=0)
        #     print(idx, embedding.shape)

        #     embedding = embedding[idx]
        #     print(idx, embedding.shape)

        item['image'] = torch.tensor(resized_image).float()  # torch.Size([5, 448, 448]) 
        # item['embedding'] = torch.tensor(embedding).float()  # torch.Size([672])
        # print(item['embedding'].shape, item['image'].shape)
        
        encoded_labels = self.CSV_file.loc[idx, 'encoded_labels']
        item['class'] = torch.tensor(encoded_labels).long()

        return item


    def __len__(self):
        return self.CSV_file.shape[0]




