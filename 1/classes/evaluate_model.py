import os
import pandas as pd
from services import evaluate_and_save
        
def evaluate_all_models():
    tasks = ["super_resolution"]
    num_epochs = [5, 10, 15, 20]
    learning_rate = [1e-4, 5e-4, 1e-3]
    criterions = ['MSELoss', 'L1Loss']
    df_results = []
    
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
                        result = evaluate_and_save(os.path.join(model_path, last_model), task, criterion, lr)
                        df_results.append(result)
                        
    if df_results:
        df = pd.DataFrame(df_results)
        
        eval_dir = os.path.join(base_dir, "outputs", "eval_results")
        os.makedirs(eval_dir, exist_ok=True)
        
        csv_path = os.path.join(eval_dir, "all_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n======================================")
        print(f"Results saved to: {csv_path}")
        print(f"======================================")
        print(df)
    else:
        print("Results list is empty. No metrics to save.")   


if __name__ == "__main__":
    evaluate_all_models()