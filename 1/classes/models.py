import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x_up = self.up1(x2)
        out = self.dec1(x_up)
        return out