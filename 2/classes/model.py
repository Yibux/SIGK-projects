import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ExposureUNet(nn.Module):
    def __init__(self):
        super(ExposureUNet, self).__init__()
        
        self.inc = DoubleConv(3, 64)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        self.out_under = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.out_over = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        u1 = self.up1(x5)
        if u1.shape != x4.shape:
            u1 = F.interpolate(u1, size=x4.shape[2:], mode='bilinear', align_corners=True)
        u1 = torch.cat([x4, u1], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        if u2.shape != x3.shape:
            u2 = F.interpolate(u2, size=x3.shape[2:], mode='bilinear', align_corners=True)
        u2 = torch.cat([x3, u2], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        if u3.shape != x2.shape:
            u3 = F.interpolate(u3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        u3 = torch.cat([x2, u3], dim=1)
        u3 = self.conv_up3(u3)
        
        u4 = self.up4(u3)
        if u4.shape != x1.shape:
            u4 = F.interpolate(u4, size=x1.shape[2:], mode='bilinear', align_corners=True)
        u4 = torch.cat([x1, u4], dim=1)
        u4 = self.conv_up4(u4)
        
        under_img = self.out_under(u4)
        over_img = self.out_over(u4)
        
        return under_img, over_img