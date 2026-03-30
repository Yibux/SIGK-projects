import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from dataset import HDREyeDataset
from model import ExposureUNet
from constants import DATASET_ROOT, OUTPUT_DIR_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trenuję na urządzeniu: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    all_scenes = os.listdir(DATASET_ROOT) if os.path.exists(DATASET_ROOT) else []
    test_scenes = [f"C{i}" for i in range(40, 47)]
    train_scenes = [scene for scene in all_scenes if scene not in test_scenes]
    
    if not train_scenes:
        print("UWAGA: Nie znaleziono folderu z danymi. Używam atrapy do testów.")
        train_scenes = ["mock_scene_1", "mock_scene_2"]

    train_dataset = HDREyeDataset(root_dir=DATASET_ROOT, scene_list=train_scenes, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = ExposureUNet().to(device)
    
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Rozpoczynam trening...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets_under = batch['target_under'].to(device) # Docelowe -2.7 EV [cite: 14]
            targets_over = batch['target_over'].to(device)   # Docelowe +2.7 EV [cite: 14]

            # Wyzerowanie gradientów
            optimizer.zero_grad()

            # Forward pass (przepuszczenie przez sieć)
            out_under, out_over = model(inputs)

            # Obliczenie straty dla obu wyjść
            loss_under = criterion(out_under, targets_under)
            loss_over = criterion(out_over, targets_over)
            
            # Całkowita strata to suma błędów z obu obrazów
            loss = loss_under + loss_over

            # Backward pass (propagacja wsteczna) i krok optymalizatora
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoka [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"--- Koniec Epoki {epoch+1} | Średnia strata: {epoch_loss:.4f} ---")

    torch.save(model.state_dict(), f"{OUTPUT_DIR_PATH}/exposure_unet_{EPOCHS}.pth")
    print(f"Model zapisany pomyślnie jako '{OUTPUT_DIR_PATH}/exposure_unet_{EPOCHS}.pth'!")

if __name__ == "__main__":
    train()