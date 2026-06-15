import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import trimesh
from torch.utils.data import DataLoader

from config import *
from dataset import MeshTransformationDataset
from models import VectorFieldDGCNN

def get_mesh_path(directory, filename):
    """Szuka pliku modelu w formatach .obj lub .ply"""
    for ext in ['.obj', '.ply']:
        
        path = os.path.join(directory, f"{filename}{ext}")
        print(f"Sprawdzanie: {path}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Nie znaleziono pliku {filename} (.obj lub .ply) w katalogu {directory}!")

def chamfer_distance_loss(p1, p2):
    """Oblicza odległość Chamfera dla dwóch chmur punktów."""
    p1 = p1.unsqueeze(2)
    p2 = p2.unsqueeze(1)
    
    dist = torch.sum((p1 - p2)**2, dim=-1)
    min_p1_to_p2, _ = torch.min(dist, dim=2)
    min_p2_to_p1, _ = torch.min(dist, dim=1)
    
    return torch.mean(min_p1_to_p2) + torch.mean(min_p2_to_p1)

def calculate_metrics_voxel(points_pred, points_target, grid_size=32):
    """Zwraca metryki IoU (Jaccard) oraz Dice na zdykretyzowanej siatce 3D (wokselizacja)."""
    points_pred = points_pred.detach().cpu().numpy()[0]
    points_target = points_target.detach().cpu().numpy()[0]
    
    all_points = np.concatenate([points_pred, points_target], axis=0)
    min_b, max_b = np.min(all_points, axis=0), np.max(all_points, axis=0)
    
    def voxelize(pts):
        pts_norm = (pts - min_b) / (max_b - min_b + 1e-8)
        indices = np.clip((pts_norm * (grid_size - 1)).astype(int), 0, grid_size - 1)
        grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        return grid
        
    grid_pred = voxelize(points_pred)
    grid_target = voxelize(points_target)
    
    intersection = np.logical_and(grid_pred, grid_target).sum()
    union = np.logical_or(grid_pred, grid_target).sum()
    volume_sum = grid_pred.sum() + grid_target.sum()
    
    iou = intersection / (union + 1e-8)
    dice = (2. * intersection) / (volume_sum + 1e-8)
    return iou, dice

def laplacian_smoothing_loss(points, k=8):
    B, N, C = points.shape
    
    inner = -2 * torch.matmul(points, points.transpose(2, 1))
    xx = torch.sum(points**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    _, idx = pairwise_distance.topk(k=k+1, dim=-1) 
    idx = idx[:, :, 1:] # [B, N, k]
    
    points_expanded = points.unsqueeze(2).expand(B, N, k, C)
    
    idx_flat = idx.reshape(B, -1)
    batch_idx = torch.arange(B).unsqueeze(1).expand(-1, N * k).reshape(B, -1)
    neighbors = points[batch_idx, idx_flat, :].reshape(B, N, k, C)
    
    neighbors_mean = neighbors.mean(dim=2) 
    
    loss_lap = torch.mean((points - neighbors_mean)**2)
    return loss_lap

def combined_loss(source_pts, pred_pts, target_pts, displacement):
    loss_cd = chamfer_distance_loss(pred_pts, target_pts)
    
    loss_lap = laplacian_smoothing_loss(pred_pts, k=8)
    
    loss_reg = torch.mean(displacement**2)
    
    total_loss = loss_cd + 0.05 * loss_lap + 0.02 * loss_reg
    return total_loss, loss_cd.item()


def train_vector_field(source_name, target_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Trening pola wektorowego (DGCNN): {source_name} -> {target_name} ---")
    
    model = VectorFieldDGCNN(k=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    source_path = get_mesh_path(DATA_DIR, source_name)
    target_path = get_mesh_path(DATA_DIR, target_name)
    
    dataset = MeshTransformationDataset(source_path, target_path, NUM_POINTS, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss_epoch = 0
        loop = tqdm(dataloader, leave=False, desc=f"Epoka {epoch+1}/{NUM_EPOCHS}")
        
        for source_pts, target_pts in loop:
            source_pts = source_pts.to(device)
            target_pts = target_pts.to(device)
            
            source_pts_t = source_pts.transpose(1, 2)
            
            optimizer.zero_grad()
            displacement = model(source_pts_t) # [B, 3, N]
            
            pred_pts = source_pts + displacement.transpose(1, 2)
            
            # Nowa funkcja straty
            loss, loss_cd_raw = combined_loss(source_pts, pred_pts, target_pts, displacement)
            
            loss.backward()
            optimizer.step()
            
            total_loss_epoch += loss.item()
            loop.set_postfix(TotalLoss=loss.item(), CD=loss_cd_raw)
            
    model_path = os.path.join(MODELS_DIR, f"{source_name}_{NUM_EPOCHS}_{LEARNING_RATE}_{NUM_POINTS}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model zapisany do: {model_path}\n")
    return model

def evaluate_and_visualize(model, source_name, target_name, is_cross_test=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    source_path = os.path.join(DATA_DIR, f"{source_name}.obj")
    target_path = os.path.join(DATA_DIR, f"{TARGET_OBJECT}.obj")
    
    dataset = MeshTransformationDataset(source_path, target_path, NUM_POINTS, augment=False, epoch_multiplier=1)
    source_pts, target_pts = dataset[0]
    
    source_pts = source_pts.unsqueeze(0).to(device)
    target_pts = target_pts.unsqueeze(0).to(device)
    
    with torch.no_grad():
        source_pts_t = source_pts.transpose(1, 2)
        displacement = model(source_pts_t)
        pred_pts = source_pts + displacement.transpose(1, 2)
        
        chamfer = chamfer_distance_loss(pred_pts, target_pts).item()
        iou, dice = calculate_metrics_voxel(pred_pts, target_pts)
        
    steps = [0.0, 0.5, 1.0]
    prefix = f"{source_name}_asian_dragon" if is_cross_test else f"{source_name}_flow"
    
    for t in steps:
        step_pts = (source_pts + t * displacement.transpose(1, 2)).detach().cpu().numpy()[0]
        pc = trimesh.PointCloud(step_pts)
        pc.export(os.path.join(SAMPLES_DIR, f"{prefix}_step_{t}.obj"))
        
    return {"Metoda": prefix, "IoU": round(iou, 4), "Dice": round(dice, 4), "Chamfer": round(chamfer, 4)}

def run_pipeline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    results = []
    trained_models = {}
    
    for obj in OBJECTS:
        model = train_vector_field(obj, TARGET_OBJECT)
        trained_models[obj] = model
        metrics = evaluate_and_visualize(model, obj, TARGET_OBJECT)
        results.append(metrics)
        
    for obj in OBJECTS:
        model = trained_models[obj]
        metrics = evaluate_and_visualize(model, TEST_OBJECT, TARGET_OBJECT, is_cross_test=True)
        metrics["Metoda"] = f"{obj}_flow_{TEST_OBJECT}"
        results.append(metrics)
        
    df = pd.DataFrame(results)
    df.to_csv(CSV_FILE, index=False)
    print(f"\nGenerowanie metryk zakończone. Wyniki w: {CSV_FILE}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_pipeline()