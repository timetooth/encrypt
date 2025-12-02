import torch
from torch import nn

class ResBlockSE(nn.Module):
    """
    Residual block with:
      - 2x Conv-BN-Act
      - Channel attention (SE)
      - Spatial attention (CBAM-style)
    Drop-in replacement for previous ResBlockSE.
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()

        # ---- main conv path ----
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.SiLU(inplace=True)  # a bit nicer than ReLU sometimes

        # ---- channel attention (SE) ----
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(1, out_channels // reduction)

        self.ca_fc1 = nn.Conv2d(out_channels, mid_channels, kernel_size=1, bias=True)
        self.ca_fc2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True)
        self.ca_sigmoid = nn.Sigmoid()

        # ---- spatial attention ----
        # We compute avg & max along channel dimension, concat -> [B, 2, H, W]
        # then conv 7x7 -> spatial attention map
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sa_sigmoid = nn.Sigmoid()

        # ---- projection for residual if channel mismatch ----
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x

        # main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # -------- Channel Attention (SE) --------
        w = self.global_pool(out)               # [B, C, 1, 1]
        w = self.ca_fc1(w)
        w = self.act(w)
        w = self.ca_fc2(w)
        w = self.ca_sigmoid(w)                  # [B, C, 1, 1]
        out = out * w                           # channel-wise reweighting

        # -------- Spatial Attention --------
        avg_map = torch.mean(out, dim=1, keepdim=True)        # [B, 1, H, W]
        max_map, _ = torch.max(out, dim=1, keepdim=True)      # [B, 1, H, W]
        sa_input = torch.cat([avg_map, max_map], dim=1)       # [B, 2, H, W]

        s = self.sa_conv(sa_input)                            # [B, 1, H, W]
        s = self.sa_sigmoid(s)
        out = out * s                                         # spatial reweighting

        # -------- Residual add --------
        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.act(out)
        return out


class ResAutoencoderXLSE(nn.Module):
    """
    Deep + wide SE-Residual autoencoder.
    """
    def __init__(self,
                 in_channels=1,
                 base_channels=64,
                 blocks_per_stage=(2, 2, 3, 3, 4)):
        super().__init__()

        b1, b2, b3, b4, bb = blocks_per_stage

        C1 = base_channels          # 64
        C2 = base_channels * 2      # 128
        C3 = base_channels * 4      # 256
        C4 = base_channels * 8      # 512
        C5 = base_channels * 16     # 1024

        # ---------- Encoder ----------
        # Stage 1: 512x512, 1 -> C1
        enc1_blocks = []
        for i in range(b1):
            in_c = in_channels if i == 0 else C1
            enc1_blocks.append(ResBlockSE(in_c, C1))
        self.enc1 = nn.Sequential(*enc1_blocks)
        self.pool1 = nn.MaxPool2d(2)   # 512 -> 256

        # Stage 2: 256x256, C1 -> C2
        enc2_blocks = []
        for i in range(b2):
            in_c = C1 if i == 0 else C2
            enc2_blocks.append(ResBlockSE(in_c, C2))
        self.enc2 = nn.Sequential(*enc2_blocks)
        self.pool2 = nn.MaxPool2d(2)   # 256 -> 128

        # Stage 3: 128x128, C2 -> C3
        enc3_blocks = []
        for i in range(b3):
            in_c = C2 if i == 0 else C3
            enc3_blocks.append(ResBlockSE(in_c, C3))
        self.enc3 = nn.Sequential(*enc3_blocks)
        self.pool3 = nn.MaxPool2d(2)   # 128 -> 64

        # Stage 4: 64x64, C3 -> C4
        enc4_blocks = []
        for i in range(b4):
            in_c = C3 if i == 0 else C4
            enc4_blocks.append(ResBlockSE(in_c, C4))
        self.enc4 = nn.Sequential(*enc4_blocks)
        self.pool4 = nn.MaxPool2d(2)   # 64 -> 32

        # ---------- Bottleneck: 32x32, C4 -> C5 ----------
        bottleneck_blocks = []
        for i in range(bb):
            in_c = C4 if i == 0 else C5
            bottleneck_blocks.append(ResBlockSE(in_c, C5))
        self.bottleneck = nn.Sequential(*bottleneck_blocks)

        # ---------- Decoder ----------
        # 32x32 -> 64x64, C5 -> C4
        self.up4 = nn.ConvTranspose2d(C5, C4, kernel_size=2, stride=2)
        dec4_blocks = [ResBlockSE(C4, C4) for _ in range(b4)]
        self.dec4 = nn.Sequential(*dec4_blocks)

        # 64x64 -> 128x128, C4 -> C3
        self.up3 = nn.ConvTranspose2d(C4, C3, kernel_size=2, stride=2)
        dec3_blocks = [ResBlockSE(C3, C3) for _ in range(b3)]
        self.dec3 = nn.Sequential(*dec3_blocks)

        # 128x128 -> 256x256, C3 -> C2
        self.up2 = nn.ConvTranspose2d(C3, C2, kernel_size=2, stride=2)
        dec2_blocks = [ResBlockSE(C2, C2) for _ in range(b2)]
        self.dec2 = nn.Sequential(*dec2_blocks)

        # 256x256 -> 512x512, C2 -> C1
        self.up1 = nn.ConvTranspose2d(C2, C1, kernel_size=2, stride=2)
        dec1_blocks = [ResBlockSE(C1, C1) for _ in range(b1)]
        self.dec1 = nn.Sequential(*dec1_blocks)

        self.final_conv = nn.Conv2d(C1, 1, kernel_size=1)

    # -------- encode/decode for latent experiments --------
    def encode(self, x):
        x = self.enc1(x)   # [B, C1, 512, 512]
        x = self.pool1(x)  # 256

        x = self.enc2(x)   # [B, C2, 256, 256]
        x = self.pool2(x)  # 128

        x = self.enc3(x)   # [B, C3, 128, 128]
        x = self.pool3(x)  # 64

        x = self.enc4(x)   # [B, C4, 64, 64]
        x = self.pool4(x)  # 32

        x = self.bottleneck(x)  # [B, C5, 32, 32]
        return x

    def decode(self, z):
        x = self.up4(z)    # [B, C4, 64, 64]
        x = self.dec4(x)

        x = self.up3(x)    # [B, C3, 128, 128]
        x = self.dec3(x)

        x = self.up2(x)    # [B, C2, 256, 256]
        x = self.dec2(x)

        x = self.up1(x)    # [B, C1, 512, 512]
        x = self.dec1(x)

        x = self.final_conv(x)
        x = torch.sigmoid(x)  # inputs assumed normalized to [0,1]
        return x

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out
