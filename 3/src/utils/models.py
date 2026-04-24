import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, condition_dim=10):
        super(Generator, self).__init__()
        
        self.input_dim = condition_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512 * 8 * 8),
            nn.BatchNorm1d(512 * 8 * 8),
            nn.ReLU(True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, condition):
        x = self.fc(condition)
        x = x.view(-1, 512, 8, 8)
        img = self.conv_blocks(x)
        return img


class Discriminator(nn.Module):
    def __init__(self, condition_dim=10):
        super(Discriminator, self).__init__()
        
        self.input_channels = 3 + condition_dim
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, img, condition):
        batch_size = img.size(0)
        
        condition_expanded = condition.view(batch_size, -1, 1, 1)
        condition_expanded = condition_expanded.repeat(1, 1, 128, 128)
  
        x = torch.cat([img, condition_expanded], dim=1)
        
        x = self.conv_blocks(x)
        validity = self.classifier(x)
        return validity