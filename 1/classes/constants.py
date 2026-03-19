from torch import nn
from models import SimpleUNet, BetterUNet, ResNetRestoration, CombinedPerceptualLoss

tasks = ["super_resolution", "denoising"]
num_epochs = [20]
learning_rate = [1e-4, 5e-4, 1e-3]
criterions = [nn.MSELoss(), nn.L1Loss(), CombinedPerceptualLoss()]
models = [SimpleUNet(), BetterUNet(), ResNetRestoration()]