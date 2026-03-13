import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

class SuperResolutionDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.low_res_options = [(32, 32), (64, 64)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        hr_image = cv2.imread(img_path)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        
        lr_size = random.choice(self.low_res_options)
        
        lr_image = cv2.resize(hr_image, lr_size, interpolation=cv2.INTER_AREA)
        
        hr_tensor = torch.from_numpy(hr_image.transpose((2, 0, 1))).float() / 255.0
        lr_tensor = torch.from_numpy(lr_image.transpose((2, 0, 1))).float() / 255.0
        
        return lr_tensor, hr_tensor