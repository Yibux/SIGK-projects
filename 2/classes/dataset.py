import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL.ExifTags import TAGS

def get_exif(image_path: str) -> dict:
    """
    Read metadata from the image.
    Interesting values: "ExposureTime", "FNumber"
    """
    image = Image.open(image_path)
    info = image._getexif()
    
    if info is None:
        raise ValueError(f"Missing metadata for image {image_path}")
        
    exif_data = {}
    for tag, value in info.items():
        decoded = TAGS.get(tag)
        if not decoded:
            continue
        exif_data[decoded] = value
        
    return exif_data

class HDREyeDataset(Dataset):
    def __init__(self, root_dir, scene_list, transform=None):
        """
        root_dir: ścieżka do folderu Bracketed_images
        scene_list: lista folderów/scen do wczytania (np. pomijając C40-C46 dla zbioru treningowego)
        transform: transformacje z torchvision.transforms (np. ToTensor, Resize)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Przechodzimy przez wybrane sceny (foldery)
        for scene in scene_list:
            scene_path = os.path.join(root_dir, scene)
            if not os.path.isdir(scene_path):
                continue
                
            # W rzeczywistości musisz zmapować nazwy plików na konkretne EV.
            # Tutaj zakładamy przykładowe nazwy plików dla danej sceny:
            input_path = os.path.join(scene_path, "0_EV.jpg") 
            under_path = os.path.join(scene_path, "minus_2.7_EV.jpg")
            over_path = os.path.join(scene_path, "plus_2.7_EV.jpg")
            
            # Sprawdzamy czy pliki istnieją
            if os.path.exists(input_path) and os.path.exists(under_path) and os.path.exists(over_path):
                self.samples.append({
                    'input': input_path,
                    'under': under_path,
                    'over': over_path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        
        # Wczytanie obrazów (RGB)
        img_input = Image.open(paths['input']).convert('RGB')
        img_under = Image.open(paths['under']).convert('RGB')
        img_over = Image.open(paths['over']).convert('RGB')
        
        # Pobranie metadanych (czasu ekspozycji) dla wejścia - przyda się później do algorytmu Debeveca
        exif_input = get_exif(paths['input'])
        exp_time = float(exif_input.get('ExposureTime', 1.0))
        
        # Transformacje (np. zamiana na Tensory PyTorcha, skalowanie)
        if self.transform:
            img_input = self.transform(img_input)
            img_under = self.transform(img_under)
            img_over = self.transform(img_over)
            
        # Zwracamy słownik z danymi
        return {
            'input': img_input,       # Tensor obrazu LDR (wejście sieci)
            'target_under': img_under, # Tensor niedoświetlonego
            'target_over': img_over,   # Tensor prześwietlonego
            'exposure_time': exp_time  # Czas ekspozycji (float)
        }