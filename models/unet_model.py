import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv2d → ReLU → Conv2d → ReLU)
    Keeps spatial size same if padding=1, kernel_size=3.
    """
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
    A standard 4-down / 4-up UNet.

    Args:
        in_channels:    input channels (e.g. 1 for X-rays)
        base_channels:  number of channels in the first layer (64 is standard)
        num_classes:    output channels (1 for binary / reconstruction)
        final_activation: "sigmoid", "tanh", or None
                          - For reconstruction in [0,1], "sigmoid" is common.
                          - For BCEWithLogitsLoss, use None and let the loss handle logits.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 1,
        final_activation: str | None = "sigmoid",
    ):
        super().__init__()

        self.final_activation = final_activation

        # ---------- Encoder ----------
        self.down1 = DoubleConv(in_channels, base_channels)          # 1 → 64
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base_channels, base_channels * 2)    # 64 → 128
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base_channels * 2, base_channels * 4)  # 128 → 256
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(base_channels * 4, base_channels * 8)  # 256 → 512
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)  # 512 → 1024

        # ---------- Decoder ----------
        # up4: 1024 → 512, then concat with skip4 (512) → 1024 → 512
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        # up3: 512 → 256, concat 256 → 512 → 256
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        # up2: 256 → 128, concat 128 → 256 → 128
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        # up1: 128 → 64, concat 64 → 128 → 64
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Final 1×1 conv to desired output channels
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    # ---------- Encoder / Decoder helpers ----------
    def encode(self, x):
        # x: [B, C, H, W]
        skip1 = self.down1(x)       # [B, base, H,   W]
        x = self.pool1(skip1)       # [B, base, H/2, W/2]

        skip2 = self.down2(x)       # [B, 2base, H/2, W/2]
        x = self.pool2(skip2)       # [B, 2base, H/4, W/4]

        skip3 = self.down3(x)       # [B, 4base, H/4, W/4]
        x = self.pool3(skip3)       # [B, 4base, H/8, W/8]

        skip4 = self.down4(x)       # [B, 8base, H/8, W/8]
        x = self.pool4(skip4)       # [B, 8base, H/16, W/16]

        bottleneck = self.bottleneck(x)  # [B, 16base, H/16, W/16]

        return bottleneck, (skip1, skip2, skip3, skip4)

    def decode(self, x, skips):
        skip1, skip2, skip3, skip4 = skips

        # up4: [B, 16base, H/16, W/16] → [B, 8base, H/8, W/8]
        x = self.up4(x)
        x = torch.cat([x, skip4], dim=1)   # [B, 16base, H/8, W/8]
        x = self.dec4(x)                   # [B, 8base, H/8, W/8]

        # up3
        x = self.up3(x)                    # [B, 4base, H/4, W/4]
        x = torch.cat([x, skip3], dim=1)   # [B, 8base, H/4, W/4]
        x = self.dec3(x)                   # [B, 4base, H/4, W/4]

        # up2
        x = self.up2(x)                    # [B, 2base, H/2, W/2]
        x = torch.cat([x, skip2], dim=1)   # [B, 4base, H/2, W/2]
        x = self.dec2(x)                   # [B, 2base, H/2, W/2]

        # up1
        x = self.up1(x)                    # [B, base, H, W]
        x = torch.cat([x, skip1], dim=1)   # [B, 2base, H, W]
        x = self.dec1(x)                   # [B, base, H, W]

        x = self.final_conv(x)             # [B, num_classes, H, W]

        if self.final_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.final_activation == "tanh":
            x = torch.tanh(x)
        # else: return raw logits

        return x

    def forward(self, x):
        bottleneck, skips = self.encode(x)
        out = self.decode(bottleneck, skips)
        return out


if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example: grayscale X-ray, reconstruction in [0,1]
    model = UNet(in_channels=1, base_channels=64, num_classes=1, final_activation="sigmoid").to(device)

    x = torch.randn(34, 1, 512, 512, device=device)
    print(f"Input shape:  {x.shape}")

    start = time.time()
    y = model(x)
    end = time.time()

    print(f"Output shape: {y.shape}")
    print(f"Inference time for batch of 34: {end - start:.4f} seconds")
