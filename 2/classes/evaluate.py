import os
import cv2
import torch
import lpips
import numpy as np
from torchvision import transforms
import pandas as pd

from model import ExposureUNet
from dataset import HDREyeDataset
from utils import measure_ev_range, read_hdr
from constants import (
    DATASET_ROOT, OUTPUT_DIR_PATH, HDR_ORIGINAL_ROOT, 
    SAMPLES_LABELS_INPUT, SAMPLES_LABELS_TARGET_UNDER, SAMPLES_LABELS_TARGET_OVER
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def tensor_to_cv2(tensor):
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def evaluate(loss_fn='L1_VGG', size=(256, 256), epochs=200, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_name = f"run_{loss_fn}_{size[0]}x{size[1]}_E{epochs}_LR{learning_rate}"
    run_dir = os.path.join(OUTPUT_DIR_PATH, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    model_name = f"exposure_unet_{epochs}_{size}_lr{learning_rate}_{loss_fn}.pth"
    model_path = os.path.join(OUTPUT_DIR_PATH, model_name)
    
    if not os.path.exists(model_path):
        print(f"Model nie znaleziony: {model_path}. Pomijam tę konfigurację.")
        return

    model = ExposureUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    eval_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    
    test_scenes = [f"C{i}" for i in range(40, 47)]
    test_dataset = HDREyeDataset(root_dir=DATASET_ROOT, scene_list=test_scenes, transform=eval_transform)
    
    psnr_under_list, lpips_under_list = [], []
    psnr_over_list, lpips_over_list = [], []
    dr_results = {}

    print(f"\nStart ewaluacji: {run_name}")
    
    for idx in range(len(test_dataset)):
        sample_meta = test_dataset.samples[idx]
        scene = sample_meta['scene']
        time_under, time_input, time_over = sample_meta['times']
        
        tensors = test_dataset[idx]
        input_tensor = tensors[SAMPLES_LABELS_INPUT].unsqueeze(0).to(device)
        gt_under_tensor = tensors[SAMPLES_LABELS_TARGET_UNDER].unsqueeze(0).to(device) * 2 - 1
        gt_over_tensor = tensors[SAMPLES_LABELS_TARGET_OVER].unsqueeze(0).to(device) * 2 - 1

        gt_under_cv2 = cv2.resize(cv2.cvtColor(cv2.imread(sample_meta[SAMPLES_LABELS_TARGET_UNDER]), cv2.COLOR_BGR2RGB), size)
        gt_over_cv2 = cv2.resize(cv2.cvtColor(cv2.imread(sample_meta[SAMPLES_LABELS_TARGET_OVER]), cv2.COLOR_BGR2RGB), size)
        input_cv2 = cv2.resize(cv2.cvtColor(cv2.imread(sample_meta[SAMPLES_LABELS_INPUT]), cv2.COLOR_BGR2RGB), size)

        with torch.no_grad():
            out_under, out_over = model(input_tensor)
            
        out_under_cv2 = tensor_to_cv2(out_under)
        out_over_cv2 = tensor_to_cv2(out_over)
        
        psnr_u = calculate_psnr(gt_under_cv2, out_under_cv2)
        psnr_o = calculate_psnr(gt_over_cv2, out_over_cv2)
        lpips_u = loss_fn_vgg(out_under * 2 - 1, gt_under_tensor).item()
        lpips_o = loss_fn_vgg(out_over * 2 - 1, gt_over_tensor).item()
        
        psnr_under_list.append(psnr_u); psnr_over_list.append(psnr_o)
        lpips_under_list.append(lpips_u); lpips_over_list.append(lpips_o)

        times = np.array([time_under, time_input, time_over], dtype=np.float32)
        gt_images_list = [gt_under_cv2, input_cv2, gt_over_cv2]
        calibrate = cv2.createCalibrateDebevec()
        crf_real = calibrate.process(gt_images_list, times)
        
        merge = cv2.createMergeDebevec()
        generated_hdr = merge.process([out_under_cv2, input_cv2, out_over_cv2], times, crf_real)

        original_hdr_path = os.path.join(HDR_ORIGINAL_ROOT, f"{scene}_HDR.hdr")
        if not os.path.exists(original_hdr_path):
            original_hdr_path = os.path.join(HDR_ORIGINAL_ROOT, f"{scene}.hdr")
        
        orig_hdr = read_hdr(original_hdr_path) if os.path.exists(original_hdr_path) else None
        dr_orig = measure_ev_range(orig_hdr) if orig_hdr is not None else 0.0
        dr_new = measure_ev_range(generated_hdr)
        
        dr_results[scene] = {
            'orig': dr_orig, 'new': dr_new, 
            'psnr_under': psnr_u, 'psnr_over': psnr_o,
            'lpips_under': lpips_u, 'lpips_over': lpips_o
        }
        
        scene_out_dir = os.path.join(run_dir, scene)
        os.makedirs(scene_out_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(scene_out_dir, "gen_under.jpg"), cv2.cvtColor(out_under_cv2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(scene_out_dir, "gen_over.jpg"), cv2.cvtColor(out_over_cv2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(scene_out_dir, "gen_hdr.hdr"), cv2.cvtColor(generated_hdr, cv2.COLOR_RGB2BGR))

    results_df = pd.DataFrame({
        'Scene': list(dr_results.keys()),
        'DR_Orig': [dr_results[s]['orig'] for s in dr_results],
        'DR_New': [dr_results[s]['new'] for s in dr_results],
        'PSNR_U': [dr_results[s]['psnr_under'] for s in dr_results],
        'PSNR_O': [dr_results[s]['psnr_over'] for s in dr_results],
        'LPIPS_U': [dr_results[s]['lpips_under'] for s in dr_results],
        'LPIPS_O': [dr_results[s]['lpips_over'] for s in dr_results],
    })
    print(results_df)
    results_df.to_csv(os.path.join(run_dir, "output.csv"), index=False)
    print(f"Wyniki zapisane w: {run_dir}")

if __name__ == "__main__":
    evaluate()