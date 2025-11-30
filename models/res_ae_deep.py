import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        return 
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.proj is not None:
            identity = self.proj(identity)
        
        out = x + identity
        out = self.relu(out)
        return out

class ResAE(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(ResAE, self).__init__()

        self.enc1 = nn.Sequential(
            ResBlock(in_channels, base_channels),
            ResBlock(base_channels, base_channels)
        )
        self.mp1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ResBlock(base_channels,base_channels*2),
            ResBlock(base_channels*2,base_channels*2)
        )
        self.mp2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ResBlock(base_channels*2, base_channels*4),
            ResBlock(base_channels*4, base_channels*4)
        )
        self.mp3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ResBlock(base_channels*4, base_channels*8),
            ResBlock(base_channels*8, base_channels*8)
        )
        self.mp4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResBlock(base_channels*8, base_channels*16),
            ResBlock(base_channels*16, base_channels*16),
        )

        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            ResBlock(base_channels*8, base_channels*8),
            ResBlock(base_channels*8, base_channels*8)
        )

        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ResBlock(base_channels*4, base_channels*4),
            ResBlock(base_channels*4, base_channels*4)
        )

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ResBlock(base_channels*2, base_channels*2),
            ResBlock(base_channels*2, base_channels*2)
        )

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ResBlock(base_channels, base_channels),
            ResBlock(base_channels, base_channels)
        )

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def encode(self, x):
        x1 = self.enc1(x)
        x1 = self.mp1(x1)

        x2 = self.enc2(x1)
        x2 = self.mp2(x2)

        x3 = self.enc3(x2)
        x3 = self.mp3(x3)

        x4 = self.enc4(x3)
        x4 = self.mp4(x4)

        bottleneck = self.bottleneck(x4)
        return bottleneck
    
    def decode(self, x):
        x1 = self.up4(x)
        x1 = self.dec4(x1)

        x2 = self.up3(x1)
        x2 = self.dec3(x2)

        x3 = self.up2(x2)
        x3 = self.dec2(x3)

        x4 = self.up1(x3)
        x4 = self.dec1(x4)

        out = self.final_conv(x4)
        out = torch.sigmoid(out)
        return out 
    def forward(self, x):
        bottleneck = self.encode(x)
        out = self.decode(bottleneck)
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ResAE(in_channels=1, base_channels=64).to(device)
    x = torch.randn((2,1,512,512)).to(device)
    y = model(x)
    print(y.shape)