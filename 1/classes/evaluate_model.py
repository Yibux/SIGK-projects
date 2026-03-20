import os
import pandas as pd
from services import evaluate_and_save

from constants import tasks, num_epochs, learning_rate, criterions, models

def evaluate_all_models():
    df_results = []
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    
    if not os.path.exists(outputs_dir):
        print("No output folder found.")
        return

    crit_names = [c if isinstance(c, str) else type(c).__name__ for c in criterions]
    lr_strings = [str(lr) for lr in learning_rate]

    for folder_name in os.listdir(outputs_dir):
        if folder_name == "eval_results":
            continue
            
        folder_path = os.path.join(outputs_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        folder_tokens = folder_name.split('_')
        
        current_task = next((t for t in tasks if t in folder_name), "unknown_task")
        current_crit = next((c for c in crit_names if c in folder_tokens), "unknown_criterion")
        current_lr = next((lr for lr in lr_strings if lr in folder_tokens), "unknown_lr")

        pth_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
        
        valid_models = []
        for f in pth_files:
            try:
                epoch_num = int(f.split('_')[3])
                if epoch_num in num_epochs:
                    valid_models.append((epoch_num, f))
            except (IndexError, ValueError):
                continue
        
        if valid_models:
            best_epoch, best_model_file = max(valid_models, key=lambda x: x[0])
            model_str = best_model_file.split('_')[0]
            model = models[0] if model_str == "SimpleUNet" else models[1] if model_str == "BetterUNet" else models[2]
                        
            print(f"\nEvaluating best model for {current_task} | Criterion: {current_crit} | LR: {current_lr} | Model: {model} | Epoch: {best_epoch}")
            model_full_path = os.path.join(folder_path, best_model_file)
            result = evaluate_and_save(model_full_path, current_task, current_crit, current_lr, model)
            
            if result:
                df_results.append(result)

    if df_results:
        df = pd.DataFrame(df_results)
        eval_dir = os.path.join(outputs_dir, "eval_results")
        os.makedirs(eval_dir, exist_ok=True)
        
        csv_path = os.path.join(eval_dir, "all_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*40}")
        print(f"Results saved to: {csv_path}")
        print(f"{'='*40}")
        print(df)
    else:
        print("No models evaluated. Make sure models match epochs in constants.")

def evaluate_single_model(model_path, task, criterion, lr, model):
    print(f"\nEvaluating model: {model_path} | Task: {task} | Criterion: {criterion} | LR: {lr} | Model: {model}")
    return evaluate_and_save(model_path, task, criterion, lr, model)

if __name__ == "__main__":
    evaluate_all_models()