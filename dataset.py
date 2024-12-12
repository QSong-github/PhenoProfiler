import os
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder

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
            img_path = os.path.join(self.image_path, source, 'images', img_name.replace('tif', 'png'))
        else:
            img_path = os.path.join(self.image_path, img_name.replace('tif', 'png'))
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
                self.CSV_file.loc[idx, 'Source'], 
                'embedding', 
                str(self.CSV_file.loc[idx, 'Metadata_Plate']), 
                str(self.CSV_file.loc[idx, 'Metadata_Well']), 
                str(self.CSV_file.loc[idx, 'Metadata_Site']), 
                'embedding.npz'
            )
        else:
            embedding_path = os.path.join(
                self.embedding_path, 
                str(self.CSV_file.loc[idx, 'Metadata_Plate']), 
                str(self.CSV_file.loc[idx, 'Metadata_Well']), 
                str(self.CSV_file.loc[idx, 'Metadata_Site']), 
                'embedding.npz'
            )

        with open(embedding_path, "rb") as data:
            info = np.load(data)
            embedding = np.median(info["features"][~np.isnan(info["features"]).any(axis=1)], axis=0)

        item['image'] = resized_image_tensor.float()
        item['embedding'] = torch.tensor(embedding).float()
        item['class'] = torch.tensor(self.CSV_file.loc[idx, 'encoded_labels']).long()

        return item

    def __len__(self):
        return self.CSV_file.shape[0]