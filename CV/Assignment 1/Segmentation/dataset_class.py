from torch.utils.data import Dataset
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class_dict_path = r"Segmentation\Dataset\class_dict.csv"
class_dict_df = pd.read_csv(class_dict_path)
class_mapping = {(row["r"], row["g"], row["b"]): idx for idx, row in class_dict_df.iterrows()}
num_classes = len(class_dict_df)

image_transform = transforms.Compose([
    transforms.Resize((360, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CamVidDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames  = sorted(os.listdir(mask_dir))
    def __len__(self):
        return len(self.image_filenames)
    def _convert_mask(self, mask):
        mask = np.array(mask, dtype=np.uint8)
        label_mask = np.zeros(mask.shape[:2], dtype=np.int64)
        for rgb, class_id in class_mapping.items():
            label_mask[np.all(mask == rgb, axis=-1)] = class_id
        return torch.tensor(label_mask, dtype=torch.long)
    def __getitem__(self, idx):
        img_path  = os.path.join(self.image_dir,  self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir,   self.mask_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("RGB")
        image = image_transform(image)
        mask = mask.resize((480, 360), Image.NEAREST)
        mask_tensor = self._convert_mask(mask)
        return image, mask_tensor