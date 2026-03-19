import os
import sys

base_dir_services = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir_services)

from services import evaluate_and_save

if __name__ == "__main__":
    tasks = ["super_resolution", "denoising"]
    num_epochs = [5, 10, 15, 20]
    learning_rate = [1e-4, 5e-4, 1e-3]
    criterions = ['MSELoss', 'L1Loss']
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for task in tasks:
        for lr in learning_rate:
            for criterion in criterions:                
                model_path = os.path.join(base_dir, "outputs", f"{task}_{criterion}_{lr}")
                
                if os.path.exists(model_path):
                    files = os.listdir(model_path)
                    
                    pth_files = [f for f in files if f.endswith('.pth')]
                    
                    if pth_files:
                        last_model = sorted(pth_files, key=lambda x: int(x.split('_')[3]))[-1]
                        
                        print(f"Evaluating model: {last_model} for task: {task}")
                        evaluate_and_save(os.path.join(model_path, last_model), task)