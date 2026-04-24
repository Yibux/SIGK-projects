import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm
import flip_evaluator as flip

import cv2
import numpy as np
import lpips
from skimage.metrics import structural_similarity as calculate_ssim
from scipy.spatial.distance import directed_hausdorff

from utils.dataset import PhongDataset
from utils.models import Generator, Discriminator
import utils.config as cfg

class L1_ResidualLoss(nn.Module):
    def __init__(self, l1_weight=1.0, res_weight=1.0):
        super(L1_ResidualLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.res_weight = res_weight

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)

        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        loss_res_y = self.l1(pred_dy, target_dy)
        loss_res_x = self.l1(pred_dx, target_dx)
        
        loss_res = loss_res_y + loss_res_x

        return (self.l1_weight * loss_l1) + (self.res_weight * loss_res)
    
def calculate_hausdorff(img_real, img_fake):
    gray_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2GRAY)
    gray_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2GRAY)
    edges_real = cv2.Canny(gray_real, 100, 200)
    edges_fake = cv2.Canny(gray_fake, 100, 200)
    pts_real = np.column_stack(np.where(edges_real > 0))
    pts_fake = np.column_stack(np.where(edges_fake > 0))
    if len(pts_real) == 0 or len(pts_fake) == 0: return 0.0
    return max(directed_hausdorff(pts_real, pts_fake)[0], directed_hausdorff(pts_fake, pts_real)[0])

def evaluate_model(generator, test_loader, device, loss_function_name, model_name="default_model", model_path=None):
    print(f"\n--- Rozpoczynam Ewaluację na zbiorze testowym ---")
    
    if model_path is not None:
        if os.path.exists(model_path):
            print(f"Wczytywanie wag generatora z pliku: {model_path}")
            generator.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Ostrzeżenie: Plik {model_path} nie istnieje. Używam obecnych wag modelu.")
    
    loss_fn_vgg = lpips.LPIPS(net='alex').to(device)
    generator.eval()
    
    total_lpips = 0.0
    total_ssim = 0.0
    total_hausdorff = 0.0
    total_flip = 0.0
    
    base_output_path = os.path.join("output", model_name, "wyniki")
    real_dir = os.path.join(base_output_path, "real")
    fake_dir = os.path.join(base_output_path, "fake")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (real_img, condition) in enumerate(tqdm(test_loader, desc="Ewaluacja")):
            real_img = real_img.to(device)
            condition = condition.to(device)
            
            fake_img = generator(condition)
            
            real_norm = (real_img + 1) / 2.0
            fake_norm = (fake_img + 1) / 2.0
            
            real_img_path = os.path.join(real_dir, f"img_{i:04d}.png")
            fake_img_path = os.path.join(fake_dir, f"img_{i:04d}.png")
            
            save_image(real_norm[0], real_img_path)
            save_image(fake_norm[0], fake_img_path)
            
            # FLIP
            _, mean_flip, _ = flip.evaluate(real_img_path, fake_img_path, "LDR")
            total_flip += mean_flip
            
            # LPIPS
            total_lpips += loss_fn_vgg(fake_img, real_img).item()
            
            real_np = (real_norm[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            fake_np = (fake_norm[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # SSIM
            total_ssim += calculate_ssim(real_np, fake_np, channel_axis=2)
            # Hausdorff Distance
            total_hausdorff += calculate_hausdorff(real_np, fake_np)

    num_samples = len(test_loader)
    avg_lpips = total_lpips / num_samples
    avg_ssim = total_ssim / num_samples
    avg_hausdorff = total_hausdorff / num_samples
    avg_flip = total_flip / num_samples
    
    df = pd.DataFrame({
        'Model Name': [model_name],
        'Loss Function': [loss_function_name],
        'FLIP': [avg_flip],
        'LPIPS': [avg_lpips],
        'SSIM': [avg_ssim],
        'Hausdorff Distance': [avg_hausdorff]
    })
    
    csv_file_path = os.path.join(base_output_path, "evaluation.csv")
    df.to_csv(csv_file_path, index=False)
    
    print(f"\nEwaluacja zakończona! Wyniki zapisane w: {csv_file_path}")
    print(f"Średnie wyniki ({model_name}):\n - FLIP: {avg_flip:.4f}\n - LPIPS: {avg_lpips:.4f}\n - SSIM: {avg_ssim:.4f}\n - Hausdorff Distance: {avg_hausdorff:.4f}")

def eval_without_training(model_path, loss_function_name, model_name):
    """
    Funkcja do uruchamiania samej ewaluacji na już wytrenowanym modelu z pliku .pth.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPrzygotowuję środowisko do ewaluacji na urządzeniu: {device}")
    
    generator = Generator(condition_dim=cfg.CONDITION_DIM).to(device)
    
    dataset = PhongDataset(csv_file=cfg.CSV_FILE, root_dir=cfg.OUTPUT_PATH)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    
    generator_seed = torch.Generator().manual_seed(42)
    _, test_dataset = random_split(dataset, [train_size, test_size], generator=generator_seed)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    evaluate_model(
        generator=generator, 
        test_loader=test_loader, 
        device=device, 
        loss_function_name=loss_function_name,
        model_name=model_name,
        model_path=model_path
    )

def train(
    loss_function='L1_ResidualLoss', 
    num_epochs=None, 
    batch_size=None, 
    learning_rate=None, 
    lambda_l1=None
):
    epochs = num_epochs if num_epochs is not None else cfg.NUM_EPOCHS
    b_size = batch_size if batch_size is not None else cfg.BATCH_SIZE
    lr = learning_rate if learning_rate is not None else cfg.LEARNING_RATE
    l1_weight = lambda_l1 if lambda_l1 is not None else cfg.LAMBDA_L1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rozpoczynam trening na urządzeniu: {device}")
    print(f"Parametry: Epoki={epochs}, Batch={b_size}, LR={lr}, Loss={loss_function}, Lambda={l1_weight}")

    os.makedirs(cfg.SAMPLES_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    dataset = PhongDataset(csv_file=cfg.CSV_FILE, root_dir=cfg.OUTPUT_PATH)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    
    generator_seed = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator_seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=b_size, 
        shuffle=True, 
        drop_last=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Zbiór treningowy: {len(train_dataset)} próbek | Zbiór testowy: {len(test_dataset)} próbek")

    generator = Generator(condition_dim=cfg.CONDITION_DIM).to(device)
    discriminator = Discriminator(condition_dim=cfg.CONDITION_DIM).to(device)

    criterion_gan = nn.BCELoss() 
    
    if loss_function == 'L1Loss':
        criterion_pixel = nn.L1Loss()
    elif loss_function == 'MSELoss':
        criterion_pixel = nn.MSELoss()
    elif loss_function == 'HuberLoss':
        criterion_pixel = nn.HuberLoss()
    elif loss_function == 'L1_ResidualLoss':
        criterion_pixel = L1_ResidualLoss(l1_weight=1.0, res_weight=2.0)
    else:
        print(f"Ostrzeżenie: Nieznana funkcja {loss_function}. Używam domyślnego L1Loss.")
        criterion_pixel = nn.L1Loss()
    
    optimizer_G = optim.Adam(
        generator.parameters(), 
        lr=lr, 
        betas=(cfg.BETA_1, cfg.BETA_2)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(), 
        lr=lr, 
        betas=(cfg.BETA_1, cfg.BETA_2)
    )

    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")

        for idx, (real_imgs, conditions) in enumerate(loop):
            real_imgs = real_imgs.to(device)
            conditions = conditions.to(device)
            
            valid = torch.ones(real_imgs.size(0), 1, device=device, requires_grad=False)
            fake = torch.zeros(real_imgs.size(0), 1, device=device, requires_grad=False)

            optimizer_G.zero_grad()
            fake_imgs = generator(conditions)

            pred_fake = discriminator(fake_imgs, conditions)
            loss_gan = criterion_gan(pred_fake, valid)
            
            loss_pixel = criterion_pixel(fake_imgs, real_imgs)
            
            loss_G = loss_gan + l1_weight * loss_pixel
            
            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            pred_real = discriminator(real_imgs, conditions)
            loss_real = criterion_gan(pred_real, valid)

            pred_fake = discriminator(fake_imgs.detach(), conditions)
            loss_fake = criterion_gan(pred_fake, fake)

            loss_D = (loss_real + loss_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()

            loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())

        if epoch % 5 == 0 or epoch == epochs - 1:
            sample_fake = (fake_imgs[:16].data + 1) / 2.0
            sample_real = (real_imgs[:16].data + 1) / 2.0
            comparison = torch.cat((sample_real, sample_fake), 0)
            
            img_path = os.path.join(cfg.SAMPLES_DIR, f"epoch_{epoch}_{loss_function}.png")
            save_image(comparison, img_path, nrow=8)

    generator_name = f"generator_{loss_function}_{epochs}_{lr}_{l1_weight}.pth"
    discriminator_name = f"discriminator_{loss_function}_{epochs}_{lr}_{l1_weight}.pth"

    torch.save(generator.state_dict(), os.path.join(cfg.MODELS_DIR, generator_name))
    torch.save(discriminator.state_dict(), os.path.join(cfg.MODELS_DIR, discriminator_name))
    print(f"Trening zakończony! Modele zapisane jako:\n - {generator_name}\n - {discriminator_name}")

    generator_name_clean = f"gen_{loss_function}_{epochs}_{lr}_{l1_weight}"
    evaluate_model(generator, test_loader, device, loss_function, model_name=generator_name_clean)

if __name__ == "__main__":
    loss_function_name = 'L1_ResidualLoss'
    epochs = cfg.NUM_EPOCHS
    learning_rate = cfg.LEARNING_RATE
    lambda_l1 = cfg.LAMBDA_L1
    model_name = f"generator_{loss_function_name}_{epochs}_{learning_rate}_{lambda_l1}"
    
    eval_without_training(
        model_path=os.path.join(cfg.MODELS_DIR, f"{model_name}.pth"),
        loss_function_name=loss_function_name,
        model_name=model_name
    )
    