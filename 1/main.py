from services import train_model
from torch import nn

if __name__ == "__main__":
    tasks = ["super_resolution", "denoising"]
    num_epochs = [5, 10, 15, 20]
    learning_rate = [1e-4, 5e-4, 1e-3]
    criterions = [nn.MSELoss(), nn.L1Loss()]
    
    trained_models = []

    for task in tasks:
        for epoch in num_epochs:
            for lr in learning_rate:
                for criterion in criterions:
                    print(f"Training {task} for {epoch} epochs with learning rate {lr} and criterion {type(criterion).__name__}")
                    trained_model = train_model(task, num_epochs=epoch, learning_rate=lr, criterion=criterion)
                    trained_models.append(trained_model)