import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import time

from dataset import HDREyeDataset
from model import ExposureUNet
from constants import DATASET_ROOT, OUTPUT_DIR_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, SAMPLES_LABELS_INPUT, SAMPLES_LABELS_TARGET_UNDER, SAMPLES_LABELS_TARGET_OVER, RESIZE_DIM

def train():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # device = torch.device("cpu")
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
    
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Rozpoczynam trening...")
    for epoch in range(EPOCHS):
        time_start = time.time()
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch[SAMPLES_LABELS_INPUT].to(device)
            targets_under = batch[SAMPLES_LABELS_TARGET_UNDER].to(device)
            targets_over = batch[SAMPLES_LABELS_TARGET_OVER].to(device)

            optimizer.zero_grad()

            out_under, out_over = model(inputs)
            
            loss_under = criterion(out_under, targets_under)
            loss_over = criterion(out_over, targets_over)
            loss = loss_under + loss_over

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoka [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        time_end = time.time()
        print(f"--- Koniec Epoki {epoch+1} | Średnia strata: {epoch_loss:.4f} | Czas: {time_end - time_start:.2f}s ---")

        if not os.path.exists(OUTPUT_DIR_PATH):
            os.makedirs(OUTPUT_DIR_PATH)
        output_model_path = f"{OUTPUT_DIR_PATH}/exposure_unet_{EPOCHS}_{RESIZE_DIM}_{LEARNING_RATE}_{epoch+1}_{epoch_loss:.4f}.pth"
        torch.save(model.state_dict(), output_model_path)
        print(f"Model zapisany pomyślnie jako '{output_model_path}'!")

if __name__ == "__main__":
    train()