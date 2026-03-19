import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from skimage.util import random_noise

class DenoisingDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.input_size = (256, 256)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        clean_image = cv2.imread(img_path)
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
        
        clean_image = cv2.resize(clean_image, self.input_size, interpolation=cv2.INTER_CUBIC)
        
        clean_image_normalized = clean_image.astype(np.float32) / 255.0
        
        sigma = random.uniform(0.01, 0.03)
        
        noisy_image = random_noise(clean_image_normalized, mode='gaussian', var=sigma**2)
        noisy_image = noisy_image.astype(np.float32) 
        
        clean_tensor = torch.from_numpy(clean_image_normalized.transpose((2, 0, 1)))
        noisy_tensor = torch.from_numpy(noisy_image.transpose((2, 0, 1)))
        
        return noisy_tensor, clean_tensor

class SuperResolutionDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.low_res_options = [(32, 32), (64, 64)]
        self.input_size = (256, 256)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        hr_image = cv2.imread(img_path)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        
        hr_image = cv2.resize(hr_image, self.input_size, interpolation=cv2.INTER_CUBIC)
        
        lr_size = random.choice(self.low_res_options)
        lr_image = cv2.resize(hr_image, lr_size, interpolation=cv2.INTER_AREA)
        # TODO: Used interpolation must be described in docs
        
        lr_image_input = cv2.resize(lr_image, self.input_size, interpolation=cv2.INTER_CUBIC)
        
        hr_tensor = torch.from_numpy(hr_image.transpose((2, 0, 1))).float() / 255.0
        lr_tensor = torch.from_numpy(lr_image_input.transpose((2, 0, 1))).float() / 255.0
        
        return lr_tensor, hr_tensor