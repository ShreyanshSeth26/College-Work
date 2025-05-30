import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

CLASS_MAPPING = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear': 4,
                 'dog': 5, 'people': 6, 'roe_deer': 7, 'sika_deer': 8, 'wild_boar': 9}

class RussianWildlifeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths, self.labels = [], []
        self.transform = transform

        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir) and class_name in CLASS_MAPPING:
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(CLASS_MAPPING[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
class AugmentedRussianWildlifeDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment=False):
        self.image_paths, self.labels = [], []
        self.transform = transform
        self.augment = augment

        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir) and class_name in CLASS_MAPPING:
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(CLASS_MAPPING[class_name])

        if self.augment:
            self.image_paths = self.image_paths * 4
            self.labels = self.labels * 4

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx % (len(self.image_paths) // 4)]).convert("RGB")
        label = self.labels[idx % (len(self.image_paths) // 4)]

        if self.augment:
            aug_type = idx // (len(self.image_paths) // 4)
            if aug_type == 1:
                image = self.add_gaussian_noise(image)
            elif aug_type == 2:
                angle = random.uniform(20, 60)
                image = F.rotate(image, angle)
            elif aug_type == 3:
                image = F.hflip(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def add_gaussian_noise(self, image):
        np_image = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, 0.05, np_image.shape)
        noisy_image = np.clip(np_image + noise, 0, 1) * 255
        return Image.fromarray(noisy_image.astype(np.uint8))