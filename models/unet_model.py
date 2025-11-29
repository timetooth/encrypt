import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        # 4 down folowed by maxpool -> bottleneck -> 4 up with convtranspose
        super().__init__()

        self.down1 = DoubleConv(in_channels,out_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(out_channels,out_channels*2)
        self.pool2 = nn.MaxPool2d(2)
        out_channels *= 2

        self.down3 = DoubleConv(out_channels,out_channels*2)
        self.pool3 = nn.MaxPool2d(2)
        out_channels *= 2

        self.down4 = DoubleConv(out_channels,out_channels*2)
        self.pool4 = nn.MaxPool2d(2)
        out_channels *= 2

        self.bottleneck = DoubleConv(out_channels,out_channels*2)
        out_channels *= 2 # 8 timesinitial channels, out_channels=1024

        self.up4 = nn.ConvTranspose2d(out_channels,out_channels//2,kernel_size=2,stride=2)
        self.dec4 = DoubleConv(out_channels,out_channels//2)
        out_channels //= 2

        self.up3 = nn.ConvTranspose2d(out_channels,out_channels//2,kernel_size=2,stride=2)
        self.dec3 = DoubleConv(out_channels,out_channels//2)
        out_channels //= 2

        self.up2 = nn.ConvTranspose2d(out_channels,out_channels//2,kernel_size=2,stride=2)
        self.dec2 = DoubleConv(out_channels,out_channels//2)
        out_channels //= 2

        self.up1 = nn.ConvTranspose2d(out_channels,out_channels//2,kernel_size=2,stride=2)
        self.dec1 = DoubleConv(out_channels,out_channels//2)
        out_channels //= 2

        self.final_conv = nn.Conv2d(in_channels=out_channels,
                                    out_channels=1,
                                    kernel_size=1
                                    )
        return
    
    def encode(self,x):
        skip1= self.down1(x)
        x1 = self.pool1(skip1)

        skip2 = self.down2(x1)
        x2 = self.pool2(skip2)

        skip3 = self.down3(x2)
        x3 = self.pool3(skip3)

        skip4 = self.down4(x3)
        x4 = self.pool4(skip4)

        bottleneck = self.bottleneck(x4)

        return bottleneck, (skip1, skip2, skip3, skip4)
    
    def decode(self, x, skips):
        skip1,skip2,skip3,skip4 = skips

        x = self.up4(x)
        x = torch.concat((x,skip4),dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.concat((x,skip3),dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.concat((x,skip2),dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.concat((x,skip1),dim=1)
        x = self.dec1(x)

        x = self.final_conv(x)
        x = torch.sigmoid(x)

        return x
        
    
    def forward(self,x):
        bottleneck, skips = self.encode(x)
        out = self.decode(bottleneck, skips)
        return out
    

if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=1, out_channels=64).to(device)
    x = torch.randn((34,1,512,512)).to(device)
    print(f"Input shape: {x.shape}")
    start = time.time()
    y = model(x)
    end = time.time()
    print(f"Inference time for batch of 34: {end-start:.4f} seconds")
    print(f"Output shape: {y.shape}")


        