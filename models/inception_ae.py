import torch
from torch import nn


class InceptionResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels should be divisible by 4"
        branch_channels = out_channels // 4

        # 1x1 branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # 3x3 branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # 3x3 dilated (dilation=2) branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # 5x5 branch
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # mix branches back to out_channels
        self.mix = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.act = nn.ReLU(inplace=True)

        # residual projection if channels differ
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)  # [B, out_channels, H, W]
        out = self.mix(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.act(out)
        return out


class InceptionResAE(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        # -------- Encoder --------
        self.enc1 = nn.Sequential(
            InceptionResBlock(in_channels, base_channels),
            InceptionResBlock(base_channels, base_channels),
        )
        self.pool1 = nn.MaxPool2d(2)  # 512 -> 256

        self.enc2 = nn.Sequential(
            InceptionResBlock(base_channels, base_channels * 2),
            InceptionResBlock(base_channels * 2, base_channels * 2),
        )
        self.pool2 = nn.MaxPool2d(2)  # 256 -> 128

        self.enc3 = nn.Sequential(
            InceptionResBlock(base_channels * 2, base_channels * 4),
            InceptionResBlock(base_channels * 4, base_channels * 4),
        )
        self.pool3 = nn.MaxPool2d(2)  # 128 -> 64

        self.enc4 = nn.Sequential(
            InceptionResBlock(base_channels * 4, base_channels * 8),
            InceptionResBlock(base_channels * 8, base_channels * 8),
        )
        self.pool4 = nn.MaxPool2d(2)  # 64 -> 32

        # -------- Bottleneck --------
        self.bottleneck = nn.Sequential(
            InceptionResBlock(base_channels * 8, base_channels * 16),
            InceptionResBlock(base_channels * 16, base_channels * 16),
        )

        # -------- Decoder --------
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8,
                                      kernel_size=2, stride=2)   # 32 -> 64
        self.dec4 = nn.Sequential(
            InceptionResBlock(base_channels * 8, base_channels * 8),
            InceptionResBlock(base_channels * 8, base_channels * 8),
        )

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                                      kernel_size=2, stride=2)   # 64 -> 128
        self.dec3 = nn.Sequential(
            InceptionResBlock(base_channels * 4, base_channels * 4),
            InceptionResBlock(base_channels * 4, base_channels * 4),
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                                      kernel_size=2, stride=2)   # 128 -> 256
        self.dec2 = nn.Sequential(
            InceptionResBlock(base_channels * 2, base_channels * 2),
            InceptionResBlock(base_channels * 2, base_channels * 2),
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                      kernel_size=2, stride=2)   # 256 -> 512
        self.dec1 = nn.Sequential(
            InceptionResBlock(base_channels, base_channels),
            InceptionResBlock(base_channels, base_channels),
        )

        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)

        x = self.enc2(x)
        x = self.pool2(x)

        x = self.enc3(x)
        x = self.pool3(x)

        x = self.enc4(x)
        x = self.pool4(x)

        x = self.bottleneck(x)
        return x  # latent feature map

    def decode(self, z):
        x = self.up4(z)
        x = self.dec4(x)

        x = self.up3(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        x = self.final_conv(x)
        x = torch.sigmoid(x)  # assuming X-rays normalized to [0,1]
        return x

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out


