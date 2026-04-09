import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import os
import time

from dataset import HDREyeDataset
from model import ExposureUNet
from constants import (
    DATASET_ROOT, OUTPUT_DIR_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, 
    SAMPLES_LABELS_INPUT, SAMPLES_LABELS_TARGET_UNDER, SAMPLES_LABELS_TARGET_OVER, 
    RESIZE_DIM, ACCUMULATION_STEPS, CRITERION
)

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.vgg_submodel = nn.Sequential(*list(vgg.children())[:16]).to(device)
        
        for param in self.vgg_submodel.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, generated, target):
        gen_norm = (generated - self.mean) / self.std
        tgt_norm = (target - self.mean) / self.std
        
        gen_features = self.vgg_submodel(gen_norm)
        tgt_features = self.vgg_submodel(tgt_norm)
        
        return self.criterion(gen_features, tgt_features)


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Trenuję na urządzeniu: {device}")
    
    transform = transforms.Compose([
        transforms.Resize(RESIZE_DIM),
        transforms.ToTensor()
    ])

    all_scenes = os.listdir(DATASET_ROOT) if os.path.exists(DATASET_ROOT) else []
    test_scenes = [f"C{i}" for i in range(40, 47)]
    train_scenes = [scene for scene in all_scenes if scene not in test_scenes]
    
    train_dataset = HDREyeDataset(root_dir=DATASET_ROOT, scene_list=train_scenes, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, persistent_workers=True)

    model = ExposureUNet().to(device)
    
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    
    if "VGG" in CRITERION:
        criterion_vgg = VGGLoss(device).to(device)
        vgg_weight = 0.1
        print("-> Aktywowano VGG Perceptual Loss")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Rozpoczynam trening (Kryterium: {CRITERION}, Akumulacja: {ACCUMULATION_STEPS})...")
    optimizer.zero_grad() 
    
    for epoch in range(EPOCHS):
        time_start = time.time()
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[SAMPLES_LABELS_INPUT].to(device)
            targets_under = batch[SAMPLES_LABELS_TARGET_UNDER].to(device)
            targets_over = batch[SAMPLES_LABELS_TARGET_OVER].to(device)

            out_under, out_over = model(inputs)
            
            loss_under = 0.0
            loss_over = 0.0
            
            if "MSE" in CRITERION:
                loss_under += criterion_mse(out_under, targets_under)
                loss_over += criterion_mse(out_over, targets_over)
            elif "L1" in CRITERION:
                loss_under += criterion_l1(out_under, targets_under)
                loss_over += criterion_l1(out_over, targets_over)
                
            if "VGG" in CRITERION:
                loss_under += vgg_weight * criterion_vgg(out_under, targets_under)
                loss_over += vgg_weight * criterion_vgg(out_over, targets_over)

            loss = (loss_under + loss_over) / ACCUMULATION_STEPS 

            loss.backward()
            
            running_loss += loss.item() * ACCUMULATION_STEPS

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 10 == 0:
                current_loss_display = loss.item() * ACCUMULATION_STEPS
                print(f"Epoka [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {current_loss_display:.4f}")

        epoch_loss = running_loss / len(train_loader)
        time_end = time.time()
        print(f"--- Koniec Epoki {epoch+1} | Średnia strata: {epoch_loss:.4f} | Czas: {time_end - time_start:.2f}s ---")

        if not os.path.exists(OUTPUT_DIR_PATH):
            os.makedirs(OUTPUT_DIR_PATH)
            
        # if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        output_model_path = f"{OUTPUT_DIR_PATH}/exposure_unet_{EPOCHS}_{RESIZE_DIM}_lr{LEARNING_RATE}_ep{epoch+1}_{epoch_loss:.4f}.pth"
        torch.save(model.state_dict(), output_model_path)
        print(f"Model zapisany pomyślnie jako '{output_model_path}'!")
    
    final_model_path = f"{OUTPUT_DIR_PATH}/exposure_unet_{EPOCHS}_{RESIZE_DIM}_lr{LEARNING_RATE}_{CRITERION}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Trening zakończony! Główny model zapisany jako '{final_model_path}'")

if __name__ == "__main__":
    train()