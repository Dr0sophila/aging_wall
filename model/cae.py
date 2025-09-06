import torch
import torch.nn as nn
import math


class DownBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU, with optional downsampling via stride."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                                   nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch), nn.SiLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                                   nn.SiLU(inplace=True), )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample by 2x (ConvT) -> Conv block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch), nn.SiLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                                   nn.SiLU(inplace=True), )

    def forward(self, x):
        return self.block(x)


# ---------- the autoencoder ----------

class AutoEncoder1024(nn.Module):
    """
    Convolutional autoencoder for 1024x1024 images.
    - Encoder: 7 downsamples (1024 -> 512 -> ... -> 8), then Linear to latent.
    - Decoder: Linear back to (C,8,8), then 7 upsamplings back to 1024.
    - Works with grayscale (in_channels=1) or RGB (in_channels=3).
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 32,
                 latent_dim: int = 1024):  # bottleneck size
        super().__init__()

        # Channel schedule
        ch1 = base_ch  # 32
        ch2 = base_ch * 2  # 64
        ch3 = base_ch * 4  # 128
        ch4 = base_ch * 8  # 256
        ch5 = base_ch * 8  # 256
        ch6 = base_ch * 8  # 256
        ch7 = base_ch * 8  # 256
        bottleneck_ch = base_ch * 8  # 256; spatial 8x8

        # Encoder: stride=2 in first conv of each block to downsample
        self.enc1 = DownBlock(in_channels, ch1, stride=2)  # 1024 -> 512
        self.enc2 = DownBlock(ch1, ch2, stride=2)  # 512  -> 256
        self.enc3 = DownBlock(ch2, ch3, stride=2)  # 256  -> 128
        self.enc4 = DownBlock(ch3, ch4, stride=2)  # 128  -> 64
        self.enc5 = DownBlock(ch4, ch5, stride=2)  # 64   -> 32
        self.enc6 = DownBlock(ch5, ch6, stride=2)  # 32   -> 16
        self.enc7 = DownBlock(ch6, ch7, stride=2)  # 16   -> 8

        # Flatten (256 * 8 * 8 = 16384 if base_ch=32) -> latent -> back
        enc_feat_dim = bottleneck_ch * 8 * 8
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(enc_feat_dim, latent_dim), nn.SiLU(inplace=True), )
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, enc_feat_dim), nn.SiLU(inplace=True), )

        # Decoder: upsample back to 1024
        self.dec1 = UpBlock(bottleneck_ch, ch7)  # 8  -> 16
        self.dec2 = UpBlock(ch7, ch6)  # 16 -> 32
        self.dec3 = UpBlock(ch6, ch5)  # 32 -> 64
        self.dec4 = UpBlock(ch5, ch4)  # 64 -> 128
        self.dec5 = UpBlock(ch4, ch3)  # 128 -> 256
        self.dec6 = UpBlock(ch3, ch2)  # 256 -> 512
        self.dec7 = UpBlock(ch2, ch1)  # 512 -> 1024

        # Output head: map back to in_channels; use Sigmoid for [0,1] data
        self.out = nn.Sequential(nn.Conv2d(ch1, in_channels, kernel_size=1, stride=1), nn.Sigmoid())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
            if isinstance(m, (nn.GroupNorm,)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        z = self.to_latent(x)
        return z, x

    def decode(self, z):
        x = self.from_latent(z)
        # reshape to (B, C, 8, 8)
        batch = z.size(0)
        # bottleneck_ch = base_ch*8; recover from modules
        bottleneck_ch = self.dec1.up.in_channels
        x = x.view(batch, bottleneck_ch, 8, 8)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.dec7(x)
        x = self.out(x)
        return x

    def forward(self, x):
        z, _ = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z
