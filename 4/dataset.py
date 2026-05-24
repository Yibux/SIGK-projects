import os
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

class MeshTransformationDataset(Dataset):
    def __init__(self, source_mesh_path, target_mesh_path, num_points=2048, augment=True, epoch_multiplier=100):
        self.num_points = num_points
        self.augment = augment
        self.epoch_multiplier = epoch_multiplier
        
        self.source_mesh = trimesh.load(source_mesh_path, force='mesh')
        self.target_mesh = trimesh.load(target_mesh_path, force='mesh')
        
        self.target_points, _ = trimesh.sample.sample_surface(self.target_mesh, self.num_points)
        self.target_points = self.normalize_points(self.target_points)

    def normalize_points(self, points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        return points / max_dist

    def __len__(self):
        return self.epoch_multiplier

    def __getitem__(self, idx):
        source_points, _ = trimesh.sample.sample_surface(self.source_mesh, self.num_points)
        source_points = self.normalize_points(source_points)

        if self.augment:
            rot_matrix = Rotation.random().as_matrix()
            source_points = np.dot(source_points, rot_matrix.T)
            
            scale = np.random.uniform(0.7, 1.3)
            source_points *= scale
            
            noise = np.random.normal(0, 0.01, source_points.shape)
            source_points += noise
            
            source_points = np.clip(source_points, -1.0, 1.0)

        return torch.tensor(source_points, dtype=torch.float32), torch.tensor(self.target_points, dtype=torch.float32)