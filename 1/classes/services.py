import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
import os
from datetime import datetime
import json
from PIL import Image
import numpy as np

from datasets import SuperResolutionDataset, DenoisingDataset
from models import SimpleUNet
from metrics import apply_baseline_denoising, calculate_metrics

base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "dataset"))
train_image_paths = glob(os.path.join(dataset_dir, "DIV2K_train_HR", "*.png"))
test_image_paths = glob(os.path.join(dataset_dir, "DIV2K_valid_HR", "*.png"))

def train_model(task, num_epochs=10, batch_size=10, learning_rate=1e-4, criterion=nn.MSELoss(), model = SimpleUNet(), device=torch.device("cpu")):
    dataset = SuperResolutionDataset(train_image_paths) if task == "super_resolution" else DenoisingDataset(train_image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dirname = f"{type(model).__name__}_{task}_{type(criterion).__name__}_{learning_rate}_{num_epochs}"
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for _, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()    
        
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        
        if epoch % 5 == 0:
            print(f"--- End of epoch {epoch+1}. Average loss: {epoch_loss:.4f} ---")
        
        if dirname not in [d.name for d in os.scandir('1/outputs') if d.is_dir()]:
            os.makedirs(f"1/outputs/{dirname}", exist_ok=True)
        
        torch.save(model.state_dict(), f"1/outputs/{dirname}/{type(model).__name__}_model_epoch_{epoch+1}_{task}_{epoch_loss:.4f}.pth")

def evaluate_and_save(model_path, task, criterion='', learning_rate='', model=SimpleUNet()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ocenianie na: {device}")
    
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    dataset = SuperResolutionDataset(test_image_paths) if task == "super_resolution" else DenoisingDataset(test_image_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if len(dataloader) == 0:
        print("Błąd: Nie znaleziono obrazów testowych. Przerywam ewaluację.")
        return
    
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    os.makedirs("outputs/eval_results", exist_ok=True)
    model_name = type(model).__name__
    
    print(f"Rozpoczynam ewaluację {len(dataloader)} obrazów...")
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = torch.clamp(model(inputs), 0, 1)
            
            input_img = inputs.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            pred_img = outputs.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            target_img = targets.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            metrics = calculate_metrics(target_img, pred_img)
            total_psnr += metrics["PSNR"]
            total_ssim += metrics["SSIM"]
            total_lpips += metrics["LPIPS"]
            
            if idx < 3:
                prefix = f"{task}_{model_name}_{criterion}_lr{learning_rate}_sample_{idx}"
                
                Image.fromarray((input_img * 255).astype(np.uint8)).save(f"outputs/eval_results/{prefix}_input.png")
                Image.fromarray((pred_img * 255).astype(np.uint8)).save(f"outputs/eval_results/{prefix}_pred.png")
                Image.fromarray((target_img * 255).astype(np.uint8)).save(f"outputs/eval_results/{prefix}_target.png")
    return {
        "Task": task,
        "Criterion": criterion,
        "Learning Rate": learning_rate,
        "Model File": os.path.basename(model_path),
        "Samples": len(dataloader),
        "PSNR": float(total_psnr / len(dataloader)),
        "SSIM": float(total_ssim / len(dataloader)),
        "LPIPS": float(total_lpips / len(dataloader))
    }
    
def evaluate_baselines():    
    print("Evaluation: OpenCV Bicubic Interpolation (Super-Resolution)...")
    sr_dataset = SuperResolutionDataset(test_image_paths)
    sr_metrics = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0}
    
    for i in range(len(sr_dataset)):
        lr_tensor, hr_tensor = sr_dataset[i]
        
        pred_img = lr_tensor.numpy().transpose((1, 2, 0))
        target_img = hr_tensor.numpy().transpose((1, 2, 0))
        
        metrics = calculate_metrics(target_img, pred_img)
        sr_metrics["PSNR"] += metrics["PSNR"]
        sr_metrics["SSIM"] += metrics["SSIM"]
        sr_metrics["LPIPS"] += metrics["LPIPS"]
        
    num_sr = len(sr_dataset)
    sr_results = {
        "Metoda": "bicubic_interpolation_super_resolution",
        "PSNR": sr_metrics["PSNR"] / num_sr,
        "SSIM": sr_metrics["SSIM"] / num_sr,
        "LPIPS": sr_metrics["LPIPS"] / num_sr
    }
    
    print("Evaluation: skimage denoise_bilateral (Denoising)...")
    dn_dataset = DenoisingDataset(test_image_paths)
    dn_metrics = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0}
    
    for i in range(len(dn_dataset)):
        noisy_tensor, clean_tensor = dn_dataset[i]
        
        noisy_img = noisy_tensor.numpy().transpose((1, 2, 0))
        target_img = clean_tensor.numpy().transpose((1, 2, 0))
        
        pred_img = apply_baseline_denoising(noisy_img)
        
        metrics = calculate_metrics(target_img, pred_img)
        dn_metrics["PSNR"] += metrics["PSNR"]
        dn_metrics["SSIM"] += metrics["SSIM"]
        dn_metrics["LPIPS"] += metrics["LPIPS"]
        
    num_dn = len(dn_dataset)
    dn_results = {
        "Metoda": "denoise_bilateral_skimage",
        "PSNR": dn_metrics["PSNR"] / num_dn,
        "SSIM": dn_metrics["SSIM"] / num_dn,
        "LPIPS": dn_metrics["LPIPS"] / num_dn
    }
    
    df_baselines = pd.DataFrame([sr_results, dn_results])
    
    df_baselines = df_baselines.round(4)
    
    return df_baselines