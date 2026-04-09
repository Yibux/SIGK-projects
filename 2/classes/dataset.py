import os
import glob
from torch.utils.data import Dataset
from PIL import Image

from utils import get_exif
from constants import EXPOSURE_TIMES, SAMPLES_LABELS_INPUT, SAMPLES_LABELS_TARGET_UNDER, SAMPLES_LABELS_TARGET_OVER

class HDREyeDataset(Dataset):
    def __init__(self, root_dir, scene_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for scene in scene_list:
            scene_path = os.path.join(root_dir, scene)
            print(scene_path)
            if not os.path.isdir(scene_path):
                continue
                
            jpg_files = list(set(glob.glob(os.path.join(scene_path, "*.jpg")) + glob.glob(os.path.join(scene_path, "*.JPG"))))
            
            if len(jpg_files) < 3:
                continue
                
            files_with_exposure = []
            for file_path in jpg_files:
                try:
                    exif_data = get_exif(file_path)
                    
                    if not exif_data or 'ExposureTime' not in exif_data:
                        raise ValueError("Brak klucza ExposureTime w EXIF")
                        
                    exp_time = float(exif_data.get('ExposureTime', 1.0))
                    files_with_exposure.append((file_path, exp_time, abs(0 - exp_time)))
                    
                except Exception as e:
                    filename = os.path.basename(file_path).lower()
                    
                    if "0.jpg" in filename:
                        exp_val = 0.0
                    elif "-27.jpg" in filename:
                        exp_val = -2.7
                    elif "27.jpg" in filename:
                        exp_val = 2.7
                    else:
                        continue
                    
                    files_with_exposure.append((file_path, exp_val, abs(0 - exp_val)))
                    
            
            files_with_exposure.sort(key=lambda x: x[1])
            
            if len(files_with_exposure) < 3:
                print(f"Nie można znaleźć 3 zdjęć z różnymi czasami ekspozycji w {scene}, pomijam tę scenę.")
                continue
            
            under_path = files_with_exposure[0][0]  
            over_path = files_with_exposure[-1][0]  
            
            closest_to_zero = min(files_with_exposure, key=lambda x: abs(x[2]))
            input_path = closest_to_zero[0]
            
            self.samples.append({
                SAMPLES_LABELS_INPUT: input_path,
                SAMPLES_LABELS_TARGET_UNDER: under_path,
                SAMPLES_LABELS_TARGET_OVER: over_path
            })
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        
        img_input = Image.open(paths[SAMPLES_LABELS_INPUT]).convert('RGB')
        img_under = Image.open(paths[SAMPLES_LABELS_TARGET_UNDER]).convert('RGB')
        img_over = Image.open(paths[SAMPLES_LABELS_TARGET_OVER]).convert('RGB')
        
        if self.transform:
            img_input = self.transform(img_input)
            img_under = self.transform(img_under)
            img_over = self.transform(img_over)
            
        return {
            SAMPLES_LABELS_INPUT: img_input,
            SAMPLES_LABELS_TARGET_UNDER: img_under,
            SAMPLES_LABELS_TARGET_OVER: img_over
        }
        
if __name__ == "__main__":
    ROOT_DIR = ""
    SCENE_LIST = ["C23", "C06", "C07"]
    
    dataset = HDREyeDataset(root_dir=ROOT_DIR, scene_list=SCENE_LIST)
    
    # sample = dataset.samples[0]
    # print(f"Input path: {sample[SAMPLES_LABELS_INPUT]} exposure time: {sample['exposure_time']}")
    # print(f"Underexposed path: {sample[SAMPLES_LABELS_TARGET_UNDER]} exposure time: {sample['exposure_time']}")
    # print(f"Overexposed path: {sample[SAMPLES_LABELS_TARGET_OVER]} exposure time: {sample['exposure_time']}")