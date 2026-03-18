import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
import os

from datasets import SuperResolutionDataset, DenoisingDataset
from models import SimpleUNet
train_image_paths = glob("../dataset/DIV2K_train_HR/*.png")


def train_model(task, num_epochs=10, batch_size=40, learning_rate=1e-4, criterion=nn.MSELoss()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
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
        
        dirname = f"{task}_{type(criterion).__name__}_{learning_rate}"
        if dirname not in [d.name for d in os.scandir('outputs') if d.is_dir()]:
            os.makedirs(f"outputs/{dirname}", exist_ok=True)
        
        torch.save(model.state_dict(), f"outputs/{dirname}/unet_model_epoch_{epoch+1}_{task}_{epoch_loss:.4f}.pth")