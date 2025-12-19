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
    
class SkipConvDown(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//2,kernel_size=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2,out_channels,kernel_size=1,padding=0),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.net(x)
    
class SkipConvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels,out_channels//2,kernel_size=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2,out_channels,kernel_size=1,padding=0),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.net(x)
        

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        # 4 down folowed by maxpool -> bottleneck -> 4 up with convtranspose
        super().__init__()
                                                                    # (1,256,256)
        self.down1 = DoubleConv(in_channels,base_channels)          # (64,256,256)
        self.pool1 = nn.MaxPool2d(2)                                # (64,128,128)

        self.down2 = DoubleConv(base_channels,base_channels*2)      # (128,128,128)
        self.pool2 = nn.MaxPool2d(2)                                # (128,64,64)   

        self.down3 = DoubleConv(base_channels*2,base_channels*4)    # (256,64,64)
        self.pool3 = nn.MaxPool2d(2)                                # (256,32,32)

        self.down4 = DoubleConv(base_channels*4,base_channels*8)    # (512,32,32)
        self.pool4 = nn.MaxPool2d(2)                                # (512,16,16)

        self.bottleneck = DoubleConv(base_channels*8,base_channels*16) # (1024,16,16)

        self.proj_down1 = SkipConvDown(base_channels,1) # (64,128,128) -> (1,128,128)
        self.proj_up1 = SkipConvUp(1,base_channels) # (1,128,128) -> (64,128,128)

        self.proj_down2 = SkipConvDown(base_channels*2,1) # (128,64,64) -> (1,64,64)
        self.proj_up2 = SkipConvUp(1,base_channels*2) # (1,64,64) -> (128,64,64)

        self.proj_down3 = SkipConvDown(base_channels*4,1) # (256,32,32) -> (1,32,32)
        self.proj_up3 = SkipConvUp(1,base_channels*4) # (1,32,32) -> (256,32,32)

        self.proj_down4 = SkipConvDown(base_channels*8,1) # (512,16,16) -> (1,16,16)
        self.proj_up4 = SkipConvUp(1,base_channels*8) # (1,16,16) -> (512,16,16)

        self.up4 = nn.ConvTranspose2d(base_channels*16,base_channels*8,kernel_size=2,stride=2)
        self.dec4 = DoubleConv(base_channels*16,base_channels*8)

        self.up3 = nn.ConvTranspose2d(base_channels*8,base_channels*4,kernel_size=2,stride=2)
        self.dec3 = DoubleConv(base_channels*8,base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4,base_channels*2,kernel_size=2,stride=2)
        self.dec2 = DoubleConv(base_channels*4,base_channels*2)

        self.up1 = nn.ConvTranspose2d(base_channels*2,base_channels,kernel_size=2,stride=2)
        self.dec1 = DoubleConv(base_channels*2,base_channels)
        
        self.final_conv = nn.Conv2d(in_channels=base_channels,
                                    out_channels=1,
                                    kernel_size=1
                                    )
        return
    
    def encode(self,x):
        skip1= self.down1(x)
        x1 = self.pool1(skip1)
        skip1_down = self.proj_down1(skip1)

        skip2 = self.down2(x1)
        x2 = self.pool2(skip2)
        skip2_down = self.proj_down2(skip2)

        skip3 = self.down3(x2)
        x3 = self.pool3(skip3)
        skip3_down = self.proj_down3(skip3)

        skip4 = self.down4(x3)
        x4 = self.pool4(skip4)
        skip4_down = self.proj_down4(skip4)

        bottleneck = self.bottleneck(x4)

        return bottleneck, (skip1_down, skip2_down, skip3_down, skip4_down)
    
    def decode(self, x, skips):
        skip1_down, skip2_down, skip3_down, skip4_down = skips

        # Input -> (1024,16,16)
        x = self.up4(x) # (512,32,32)
        skip4 = self.proj_up4(skip4_down) # (512,16,16) -> (512,32,32)
        x = torch.concat((x,skip4),dim=1) # (1024,32,32)
        x = self.dec4(x) # (512,32,32)

        x = self.up3(x) # (256,64,64)
        skip3 = self.proj_up3(skip3_down) # (256,32,32) -> (256,64,64)
        x = torch.concat((x,skip3),dim=1) # (512,64,64)
        x = self.dec3(x) # (256,64,64)

        x = self.up2(x) # (128,128,128)
        skip2 = self.proj_up2(skip2_down) # (128,64,64) -> (128,128,128)
        x = torch.concat((x,skip2),dim=1) # (256,128,128)
        x = self.dec2(x) # (128,128,128)

        x = self.up1(x) # (64,256,256)
        skip1 = self.proj_up1(skip1_down) # (64,128,128) -> (64,256,256)
        x = torch.concat((x,skip1),dim=1) # (128,256,256)
        x = self.dec1(x) # (64,256,256)

        x = self.final_conv(x) # (1,256,256)
        x = torch.sigmoid(x) # (1,256,256)

        return x
        
    
    def forward(self,x):
        bottleneck, skips = self.encode(x)
        out = self.decode(bottleneck, skips)
        return out
    

if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=1, base_channels=64).to(device)
    x = torch.randn((1,1,256,256)).to(device)
    print(f"Input shape: {x.shape}")
    start = time.time()
    y = model(x)
    end = time.time()
    print(f"Inference time for batch of 34: {end-start:.4f} seconds")
    print(f"Output shape: {y.shape}")


        