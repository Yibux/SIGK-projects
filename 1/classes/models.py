import lpips
import torch
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

class DoubleConv(nn.Module):
    """Blok dwóch warstw konwolucyjnych używany w U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class BetterUNet(nn.Module):
    def __init__(self):
        super(BetterUNet, self).__init__()
        
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        
        out = self.sigmoid(self.outc(x))
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class ResNetRestoration(nn.Module):
    def __init__(self, num_res_blocks=5):
        super(ResNetRestoration, self).__init__()
        
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )
        
        self.conv_output = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_input = x 
        
        out = self.conv_input(x)
        out = self.res_blocks(out)
        out = self.conv_output(out)
        
        out = original_input + out 
        
        return self.sigmoid(out)
    
    
class CombinedPerceptualLoss(nn.Module):
    def __init__(self, weight_l1=1.0, weight_perceptual=0.1):
        super(CombinedPerceptualLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval() 
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
            
        self.weight_l1 = weight_l1
        self.weight_perceptual = weight_perceptual

    def forward(self, pred, target):
        loss_l1 = self.l1_loss(pred, target)
        
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        
        loss_p = self.perceptual_loss(pred_scaled, target_scaled).mean() 
        
        return self.weight_l1 * loss_l1 + self.weight_perceptual * loss_p