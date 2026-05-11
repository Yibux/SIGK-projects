import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PhongDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Ścieżka do pliku dataset.csv.
            root_dir (string): Katalog z obrazkami.
            transform (callable, optional): Opcjonalne transformacje na obrazku.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        params = self.data_frame.iloc[idx, 1:].values.astype('float32')
        
        params[0:3] = params[0:3] / 3.0   # model_tx, model_ty, model_tz
        params[3:6] = params[3:6] / 255.0  # diffuse_r, g, b
        params[6] = (params[6] - 3.0) / (20.0 - 3.0)      # shininess
        params[7:10] = params[7:10] / 20.0 # light_px, py, pz
        
        params_tensor = torch.tensor(params)

        return image, params_tensor