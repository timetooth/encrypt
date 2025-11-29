import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    This is now really just a plain encoder–decoder conv net (no skip connections).
    Keeping the name 'UNet' is a bit misleading, but functionally it's fine.
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, base_channels)        # 1 -> 64
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base_channels, base_channels * 2)  # 64 -> 128
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base_channels * 2, base_channels * 4)  # 128 -> 256
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(base_channels * 4, base_channels * 8)  # 256 -> 512
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck: 512 -> 1024
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        # Decoder (no concatenation, so in_channels don't double)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 8, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 2, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels, base_channels)

        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def encode(self, x):
        x = self.down1(x)
        x = self.pool1(x)

        x = self.down2(x)
        x = self.pool2(x)

        x = self.down3(x)
        x = self.pool3(x)

        x = self.down4(x)
        x = self.pool4(x)

        x = self.bottleneck(x)
        return x

    def decode(self, x):
        x = self.up4(x)
        x = self.dec4(x)

        x = self.up3(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        x = self.final_conv(x)
        x = torch.sigmoid(x)  # assuming inputs are scaled to [0,1]
        return x

    def forward(self, x):
        bottleneck = self.encode(x)
        out = self.decode(bottleneck)
        return out


if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=1, base_channels=64).to(device)
    x = torch.randn((34, 1, 512, 512), device=device)
    print(f"Input shape: {x.shape}")

    start = time.time()
    y = model(x)
    end = time.time()

    print(f"Inference time for batch of 34: {end-start:.4f} seconds")
    print(f"Output shape: {y.shape}")
