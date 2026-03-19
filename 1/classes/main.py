from services import train_model
import torch
from constants import tasks, num_epochs, learning_rate, criterions, models

if __name__ == "__main__":
    
    trained_models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    for task in tasks:
        for epoch in num_epochs:
            for lr in learning_rate:
                for criterion in criterions:
                    print(f"Training {task} for {epoch} epochs with learning rate {lr} and criterion {type(criterion).__name__} and model {type(models[0]).__name__}")
                    for model in models:
                        trained_model = train_model(task, num_epochs=epoch, learning_rate=lr, criterion=criterion, model=model, device=device)
                        trained_models.append(trained_model)