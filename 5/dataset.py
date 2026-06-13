import os
import torch
import numpy as np
import pandas as pd
import bvhio
from torch.utils.data import Dataset
from config import *

class CMUMotionDataset(Dataset):
    def __init__(self, csv_index_path):
        self.csv_index_path = csv_index_path
        self.samples = []
        self.labels = []
        
        self._load_and_process_data()

    def _load_and_process_data(self):
        print(f"Ładowanie danych treningowych z indeksu: {self.csv_index_path}...")
        df = pd.read_csv(self.csv_index_path)
        
        target_joints_mapping = {
            Joint.HEAD: ['Head', 'head'],
            Joint.NECK: ['Neck', 'Neck1', 'neck'],
            Joint.PELVIS: ['Hips', 'hips', 'Root', 'Pelvis'],
            Joint.RIGHT_SHOULDER: ['RightShoulder', 'RightArm', 'RightUpArm', 'rShldr', 'RightCollar'],
            Joint.RIGHT_ELBOW: ['RightElbow', 'RightForeArm', 'RightLowArm', 'rForeArm'],
            Joint.RIGHT_WRIST: ['RightWrist', 'RightHand', 'rHand'],
            Joint.LEFT_SHOULDER: ['LeftShoulder', 'LeftArm', 'LeftUpArm', 'lShldr', 'LeftCollar'],
            Joint.LEFT_ELBOW: ['LeftElbow', 'LeftForeArm', 'LeftLowArm', 'lForeArm'],
            Joint.LEFT_WRIST: ['LeftWrist', 'LeftHand', 'lHand'],
            Joint.RIGHT_HIP: ['RightHip', 'RightUpLeg', 'rThigh'],
            Joint.RIGHT_KNEE: ['RightKnee', 'RightLeg', 'rShin'],
            Joint.RIGHT_ANKLE: ['RightAnkle', 'RightFoot', 'rFoot'],
            Joint.LEFT_HIP: ['LeftHip', 'LeftUpLeg', 'lThigh'],
            Joint.LEFT_KNEE: ['LeftKnee', 'LeftLeg', 'lShin'],
            Joint.LEFT_ANKLE: ['LeftAnkle', 'LeftFoot', 'lFoot']
        }
        
        for index, row in df.iterrows():
            file_path = row['filepath']
            action = row['action']
            action_label = ACTIONS[action]
            
            try:
                root = bvhio.readAsHierarchy(file_path)
                frames_data = []
                
                total_frames = len(root.Keyframes)
                num_frames_to_read = min(NUM_FRAMES, total_frames)
                
                for frame_idx in range(num_frames_to_read):
                    root.loadPose(frame_idx)
                    
                    joint_coords = [None] * NUM_JOINTS
                    pelvis_pos = None
                    
                    for joint_enum, possible_names in target_joints_mapping.items():
                        joint_nodes = None
                        
                        for try_name in possible_names:
                            joint_nodes = root.filter(try_name)
                            if joint_nodes:
                                break
                                
                        if not joint_nodes:
                            raise ValueError(f"Brak węzła dla {joint_enum.name}")
                                
                        pos = joint_nodes[0].PositionWorld
                        coords = np.array([pos.x, pos.y, pos.z])
                        
                        if joint_enum == Joint.PELVIS:
                            pelvis_pos = coords
                            
                        joint_coords[joint_enum.value] = coords
                        
                    joint_coords = np.array(joint_coords)
                    
                    if pelvis_pos is not None:
                        joint_coords = joint_coords - pelvis_pos
                        
                    frames_data.append(joint_coords)
                
                while len(frames_data) < NUM_FRAMES:
                    frames_data.append(frames_data[-1])
                    
                tensor_data = np.array(frames_data[:NUM_FRAMES]) 
                print(f"Loading '{index}/{len(df)-1}': {file_path} - Success")
                
                self.samples.append(tensor_data)
                self.labels.append(action_label)
                
            except Exception as e:
                continue

        print(f"Sukces! Załadowano {len(self.samples)} poprawnych animacji do treningu.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y