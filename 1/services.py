import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
import os
from datetime import datetime

from datasets import SuperResolutionDataset, DenoisingDataset
from models import SimpleUNet
from metrics import calculate_metrics
train_image_paths = glob("../dataset/DIV2K_train_HR/*.png")


def train_model(task, num_epochs=10, batch_size=40, learning_rate=1e-4, criterion=nn.MSELoss()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    startDateTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset = SuperResolutionDataset(train_image_paths) if task == "super_resolution" else DenoisingDataset(train_image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleUNet().to(device)
    criterion = criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        print(f"--- End of epoch {epoch+1}. Average loss: {epoch_loss:.4f} ---")
        
        dirname = f"{task}_{type(criterion).__name__}_{learning_rate}_{startDateTime}"
        if dirname not in [d.name for d in os.scandir('outputs') if d.is_dir()]:
            os.makedirs(f"outputs/{dirname}", exist_ok=True)
        
        torch.save(model.state_dict(), f"outputs/{dirname}/unet_model_epoch_{epoch+1}_{task}_{epoch_loss:.4f}.pth")
    
def evaluate_model_on_test_set(model, test_image_paths, task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)
    model.eval()
    
    dataset = SuperResolutionDataset(test_image_paths) if task == "super_resolution" else DenoisingDataset(test_image_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            outputs = torch.clamp(outputs, 0, 1)
            
            pred_img = outputs.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            target_img = targets.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            metrics = calculate_metrics(target_img, pred_img)
            
            total_psnr += metrics["PSNR"]
            total_ssim += metrics["SSIM"]
            total_lpips += metrics["LPIPS"]
            
            print(f"Processed image {idx+1}/{len(dataloader)}")
            
    num_samples = len(dataloader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Task: {task}")
    print(f"Average PSNR:  {avg_psnr:.4f}")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    
    return {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips
    }