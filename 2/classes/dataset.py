import os
import glob
from torch.utils.data import Dataset
from PIL import Image

from utils import get_exif

class HDREyeDataset(Dataset):
    def __init__(self, root_dir, scene_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for scene in scene_list:
            scene_path = os.path.join(root_dir, scene)
            if not os.path.isdir(scene_path):
                continue
                
            jpg_files = glob.glob(os.path.join(scene_path, "*.jpg")) + glob.glob(os.path.join(scene_path, "*.JPG"))
            
            if len(jpg_files) < 3:
                continue
                
            files_with_exposure = []
            for file_path in jpg_files:
                exif_data = get_exif(file_path)
                exp_time = float(exif_data.get('ExposureTime', 1.0))
                files_with_exposure.append((file_path, exp_time))
                
            files_with_exposure.sort(key=lambda x: x[1])
            
            under_path = files_with_exposure[0][0]  # Najkrótszy czas (-2.7 EV)
            input_path = files_with_exposure[1][0]  # Środkowy czas (0 EV)
            over_path = files_with_exposure[-1][0]  # Najdłuższy czas (+2.7 EV)
            print(files_with_exposure[0])
            
            self.samples.append({
                'input': input_path,
                'target_under': under_path,
                'target_over': over_path,
                'exposure_time': files_with_exposure[1][1]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        
        img_input = Image.open(paths['input']).convert('RGB')
        img_under = Image.open(paths['target_under']).convert('RGB')
        img_over = Image.open(paths['target_over']).convert('RGB')
        
        if self.transform:
            img_input = self.transform(img_input)
            img_under = self.transform(img_under)
            img_over = self.transform(img_over)
            
        return {
            'input': img_input,       
            'target_under': img_under, 
            'target_over': img_over,   
            'exposure_time': paths['exposure_time']
        }
        
if __name__ == "__main__":
    ROOT_DIR = "D:\\studia\\Studia 2 stopnia\\Semestr 3\\SIGK\\SIGK-projects\\dataset\\HDREye\\images\\Bracketed_images"
    SCENE_LIST = ["C05", "C06", "C07"]
    
    dataset = HDREyeDataset(root_dir=ROOT_DIR, scene_list=SCENE_LIST)
    
    # sample = dataset.samples[0]
    # print(f"Input path: {sample['input']} exposure time: {sample['exposure_time']}")
    # print(f"Underexposed path: {sample['target_under']} exposure time: {sample['exposure_time']}")
    # print(f"Overexposed path: {sample['target_over']} exposure time: {sample['exposure_time']}")