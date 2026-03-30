import glob
import os
import cv2
import torch
import lpips
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image

from model import ExposureUNet
from utils import measure_ev_range, read_hdr, get_exif
from constants import DATASET_ROOT, OUTPUT_DIR_PATH, HDR_ORIGINAL_ROOT, EPOCHS

def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def tensor_to_cv2(tensor):
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ExposureUNet().to(device)
    model.load_state_dict(torch.load(f"exposure_unet_{EPOCHS}.pth", map_location=device, weights_only=False))
    model.eval()

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    test_scenes = [f"C{i}" for i in range(40, 47)]
    
    psnr_under_list, lpips_under_list = [], []
    psnr_over_list, lpips_over_list = [], []
    
    dr_results = {}

    print("Rozpoczynam ewaluację na scenach C40-C46...")
    
    for scene in test_scenes:
        print(f"\nPrzetwarzam scenę: {scene}")
        
        scene_dir = os.path.join(DATASET_ROOT, scene)
        # Szukamy wszystkich plików JPG w folderze danej sceny
        jpg_files = glob.glob(os.path.join(scene_dir, "*.jpg")) + glob.glob(os.path.join(scene_dir, "*.JPG"))

        if len(jpg_files) < 3:
            print(f"Brak wystarczającej liczby zdjęć w {scene}, pomijam.")
            continue

        files_with_exposure = []
        for file_path in jpg_files:
            exif_data = get_exif(file_path)
            exp_time = float(exif_data.get('ExposureTime', 1.0))
            files_with_exposure.append((file_path, exp_time))

        files_with_exposure.sort(key=lambda x: x[1])

        # 1. Najkrótszy czas naświetlania -> obraz najciemniejszy (niedoświetlony, docelowo EV = -2.7)
        gt_under_path = files_with_exposure[0][0]

        # 2. Środkowy czas naświetlania -> obraz bazowy (LDR wejściowy, 0 EV)
        input_path = files_with_exposure[1][0]

        # 3. Najdłuższy czas naświetlania -> obraz najjaśniejszy (prześwietlony, docelowo EV = +2.7)
        gt_over_path = files_with_exposure[-1][0]

        original_hdr_path = os.path.join(HDR_ORIGINAL_ROOT, f"{scene}.hdr")

        print(f"Wejście: {os.path.basename(input_path)}")
        print(f"Target (-2.7 EV): {os.path.basename(gt_under_path)}")
        print(f"Target (+2.7 EV): {os.path.basename(gt_over_path)}")
        
        if not os.path.exists(input_path):
            print(f"Brak plików dla {scene}, pomijam.")
            continue

        img_input_pil = Image.open(input_path).convert('RGB')
        input_tensor = ToTensor()(img_input_pil).unsqueeze(0).to(device)
        
        gt_under_cv2 = cv2.cvtColor(cv2.imread(gt_under_path), cv2.COLOR_BGR2RGB)
        gt_over_cv2 = cv2.cvtColor(cv2.imread(gt_over_path), cv2.COLOR_BGR2RGB)
        
        gt_under_tensor = ToTensor()(Image.open(gt_under_path).convert('RGB')).unsqueeze(0).to(device) * 2 - 1
        gt_over_tensor = ToTensor()(Image.open(gt_over_path).convert('RGB')).unsqueeze(0).to(device) * 2 - 1

        with torch.no_grad():
            out_under, out_over = model(input_tensor)
            
        out_under_cv2 = tensor_to_cv2(out_under)
        out_over_cv2 = tensor_to_cv2(out_over)
        input_cv2 = tensor_to_cv2(input_tensor)

        out_under_lpips = out_under * 2 - 1
        out_over_lpips = out_over * 2 - 1

        psnr_u = calculate_psnr(gt_under_cv2, out_under_cv2)
        psnr_o = calculate_psnr(gt_over_cv2, out_over_cv2)
        
        lpips_u = loss_fn_vgg(out_under_lpips, gt_under_tensor).item()
        lpips_o = loss_fn_vgg(out_over_lpips, gt_over_tensor).item()
        
        psnr_under_list.append(psnr_u); psnr_over_list.append(psnr_o)
        lpips_under_list.append(lpips_u); lpips_over_list.append(lpips_o)

        # 4. SKŁADANIE HDR ALGORYTMEM DEBEVECA
        # Pobieranie czasów naświetlania
        exif_input = get_exif(input_path)
        exif_under = get_exif(gt_under_path)
        exif_over = get_exif(gt_over_path)
        
        times = np.array([
            float(exif_under.get('ExposureTime', 1.0)), 
            float(exif_input.get('ExposureTime', 1.0)), 
            float(exif_over.get('ExposureTime', 1.0))
        ], dtype=np.float32)

        images_list = [out_under_cv2, input_cv2, out_over_cv2] # Zauważ, że używamy WYGENEROWANYCH obrazów!

        # Odzyskiwanie funkcji odpowiedzi kamery (CRF) metodą SVD
        calibrate = cv2.createCalibrateDebevec()
        crf = calibrate.process(images_list, times)
        
        # Złączenie w HDR
        merge = cv2.createMergeDebevec()
        generated_hdr = merge.process(images_list, times, crf)

        # 5. METRYKI HDR (Dynamic Range)
        orig_hdr = read_hdr(original_hdr_path)
        
        dr_orig = measure_ev_range(orig_hdr)
        dr_new = measure_ev_range(generated_hdr)
        
        dr_results[scene] = {'orig': dr_orig, 'new': dr_new}

    # DRUKOWANIE WYNIKÓW (gotowe do wklejenia w sprawozdanie!)
    
    print("\n" + "="*50)
    print("Tabela 1: Metryki PSNR i LPIPS (Średnie dla C40-C46)")
    print("Metoda \t\t PSNR \t\t LPIPS")
    print(f"underexposed \t {np.mean(psnr_under_list):.4f} \t {np.mean(lpips_under_list):.4f}")
    print(f"overexposed \t {np.mean(psnr_over_list):.4f} \t {np.mean(lpips_over_list):.4f}")
    print("="*50)
    
    print("\n" + "="*50)
    print("Tabela 2: Dynamic Range")
    print("Obraz \t Dynamic Range Original \t Dynamic Range New")
    for scene in test_scenes:
        if scene in dr_results:
            print(f"{scene} \t {dr_results[scene]['orig']:.4f} \t\t\t {dr_results[scene]['new']:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate()