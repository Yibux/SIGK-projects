import glob
import os
import cv2
import torch
import lpips
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd

from model import ExposureUNet
from utils import measure_ev_range, read_hdr, get_exif
from constants import DATASET_ROOT, OUTPUT_DIR_PATH, HDR_ORIGINAL_ROOT, EPOCHS, RESIZE_DIM, LEARNING_RATE, OUTPUT_MODEL_NAME
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    model_path = f"{OUTPUT_DIR_PATH}/{OUTPUT_MODEL_NAME}"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    eval_transform = transforms.Compose([
        transforms.Resize(RESIZE_DIM),
        transforms.ToTensor()
    ])
    
    test_scenes = [f"C{i}" for i in range(40, 47)]
    
    psnr_under_list, lpips_under_list = [], []
    psnr_over_list, lpips_over_list = [], []
    
    dr_results = {}

    print("Rozpoczynam ewaluację na scenach C40-C46...")
    
    for scene in test_scenes:
        print(f"\nPrzetwarzam scenę: {scene}")
        
        scene_dir = os.path.join(DATASET_ROOT, scene)
        jpg_files = glob.glob(os.path.join(scene_dir, "*.jpg")) + glob.glob(os.path.join(scene_dir, "*.JPG"))

        if len(jpg_files) < 3:
            print(f"Brak wystarczającej liczby zdjęć w {scene}, pomijam.")
            continue
        
        files_with_exposure = []
        for file_path in jpg_files:
            try:
                exif_data = get_exif(file_path)
                if not exif_data or 'ExposureTime' not in exif_data:
                    raise ValueError("Brak klucza ExposureTime w EXIF")
                
                exp_time = float(exif_data.get('ExposureTime', 1.0))
                ev_val = np.log2(exp_time) if exp_time > 0 else 0.0 
                
            except Exception as e:
                filename = os.path.basename(file_path).lower()
                if "0.jpg" in filename:
                    ev_val = 0.0
                elif "-27.jpg" in filename:
                    ev_val = -2.7
                elif "27.jpg" in filename:
                    ev_val = 2.7
                else:
                    continue
                
                exp_time = 2 ** ev_val 

            files_with_exposure.append((file_path, exp_time, abs(0 - ev_val)))

        if len(files_with_exposure) < 3:
            continue
            
        files_with_exposure.sort(key=lambda x: x[1])

        gt_under_path, time_under, _ = files_with_exposure[0]
        gt_over_path, time_over, _ = files_with_exposure[-1]
        
        closest_to_zero = min(files_with_exposure, key=lambda x: x[2])
        input_path, time_input, _ = closest_to_zero

        original_hdr_path = os.path.join(HDR_ORIGINAL_ROOT, f"{scene}_HDR.hdr")
        print(f"Ścieżki: \n  Underexposed: {gt_under_path}\n  Input: {input_path}\n  Overexposed: {gt_over_path}")
        
        if not os.path.exists(input_path):
            print(f"Brak plików dla {scene}, pomijam.")
            continue

        img_input_pil = Image.open(input_path).convert('RGB')
        input_tensor = eval_transform(img_input_pil).unsqueeze(0).to(device)
        
        gt_under_tensor = eval_transform(Image.open(gt_under_path).convert('RGB')).unsqueeze(0).to(device) * 2 - 1
        gt_over_tensor = eval_transform(Image.open(gt_over_path).convert('RGB')).unsqueeze(0).to(device) * 2 - 1

        gt_under_cv2 = cv2.cvtColor(cv2.imread(gt_under_path), cv2.COLOR_BGR2RGB)
        gt_under_cv2 = cv2.resize(gt_under_cv2, RESIZE_DIM)
        
        gt_over_cv2 = cv2.cvtColor(cv2.imread(gt_over_path), cv2.COLOR_BGR2RGB)
        gt_over_cv2 = cv2.resize(gt_over_cv2, RESIZE_DIM)

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

        times = np.array([time_under, time_input, time_over], dtype=np.float32)

        images_list = [out_under_cv2, input_cv2, out_over_cv2]

        calibrate = cv2.createCalibrateDebevec()
        crf = calibrate.process(images_list, times)
        
        merge = cv2.createMergeDebevec()
        generated_hdr = merge.process(images_list, times, crf)

        orig_hdr = read_hdr(original_hdr_path)
        
        dr_orig = measure_ev_range(orig_hdr) if orig_hdr is not None else 0.0
        dr_new = measure_ev_range(generated_hdr)
        
        dr_results[scene] = {'orig': dr_orig, 'new': dr_new, 'psnr_under': psnr_u, 'psnr_over': psnr_o}
        
        del input_tensor, gt_under_tensor, gt_over_tensor, out_under, out_over, out_under_lpips, out_over_lpips
        torch.cuda.empty_cache()
        
        scene_out_dir = os.path.join(OUTPUT_DIR_PATH, scene)
        os.makedirs(scene_out_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(scene_out_dir, "gen_under.jpg"), cv2.cvtColor(out_under_cv2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(scene_out_dir, "gen_over.jpg"), cv2.cvtColor(out_over_cv2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(scene_out_dir, "gen_hdr.hdr"), generated_hdr)
    
    print("\n" + "="*50)
    print("Tabela 1: Metryki PSNR i LPIPS (Średnie dla test_scenes)")
    if psnr_under_list:
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
    
    results_df = pd.DataFrame({
        'Scene': list(dr_results.keys()),
        'Dynamic Range Original': [dr_results[scene]['orig'] for scene in dr_results],
        'Dynamic Range New': [dr_results[scene]['new'] for scene in dr_results],
        'PSNR Underexposed': [dr_results[scene]['psnr_under'] for scene in dr_results],
        'PSNR Overexposed': [dr_results[scene]['psnr_over'] for scene in dr_results],
    })
    results_df.to_csv(f"{OUTPUT_DIR_PATH}/output.csv", index=False)


if __name__ == "__main__":
    evaluate()