import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_bilateral
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

def calculate_metrics(target_img, pred_img):
    psnr_val = psnr(target_img, pred_img, data_range=1.0)
    ssim_val = ssim(target_img, pred_img, data_range=1.0, channel_axis=-1)

    target_tensor = torch.from_numpy(target_img.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    pred_tensor = torch.from_numpy(pred_img.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    
    target_tensor = target_tensor * 2.0 - 1.0
    pred_tensor = pred_tensor * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_val = loss_fn_vgg(pred_tensor, target_tensor).item()
        
    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "LPIPS": lpips_val
    }

def apply_baseline_denoising(noisy_img):
    baseline_out = denoise_bilateral(
        noisy_img, 
        channel_axis=-1,
        sigma_color=0.05,
        sigma_spatial=15
    )
    
    return baseline_out.astype(np.float32)