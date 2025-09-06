import torch.nn as nn
import os, math, torch
import torch.nn.functional as F
from torchvision import utils as vutils


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

    def __init__(self, in_channels: int = 3, base_ch: int = 32, latent_dim: int = 1024):  # bottleneck size
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
        self.bottleneck_ch = bottleneck_ch

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
        x = x.view(batch, self.bottleneck_ch, 8, 8)

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


@torch.no_grad()
def _psnr(x, y, eps=1e-8):
    # x,y in [0,1]
    mse = F.mse_loss(x, y, reduction='mean').clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


@torch.no_grad()
def _evaluate(model, loader, device, sample_dir=None, epoch=0, max_grids=2, grid_n=4):
    model.eval()
    totals = {'mse': 0.0, 'mae': 0.0, 'psnr': 0.0}
    count = 0
    saved = 0

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        rec, _ = model(imgs)

        mse = F.mse_loss(rec, imgs, reduction='mean').item()
        mae = F.l1_loss(rec, imgs, reduction='mean').item()
        p = _psnr(rec, imgs).item()

        b = imgs.size(0)
        totals['mse'] += mse * b
        totals['mae'] += mae * b
        totals['psnr'] += p * b
        count += b

        if sample_dir and saved < max_grids:
            grid = vutils.make_grid(torch.cat([imgs[:grid_n], rec[:grid_n]], dim=0), nrow=grid_n)
            os.makedirs(sample_dir, exist_ok=True)
            vutils.save_image(grid, os.path.join(sample_dir, f"val_ep{epoch:03d}_{saved:02d}.png"))
            saved += 1

    if count == 0:  # empty loader
        return {k: float("nan") for k in totals}

    return {k: v / count for k, v in totals.items()}


def train(model, train_loader, val_loader=None, *, epochs=50, lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.99),
          clip_grad=1.0, amp=True, device=None, outdir="runs/ae1024", save_every=5, resume_path="", log_interval=50,
          loss_mix=None,  # None or float in [0,1]: loss = (1-a)*MSE + a*L1
          ):
    """
    Train an autoencoder.
    - model: your AutoEncoder1024 (already constructed)
    - train_loader / val_loader: PyTorch DataLoader(s) yielding (image, label)
    - device: torch.device or 'cuda'/'cpu' (auto-detected if None)
    - loss_mix: if set to `a` (e.g., 0.3), uses loss = (1-a)*MSE + a*L1
    Returns: history dict with per-epoch metrics and the path to best checkpoint.
    """
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = os.path.join(outdir, "ckpts");
    os.makedirs(ckpt_dir, exist_ok=True)
    sample_dir = os.path.join(outdir, "samples");
    os.makedirs(sample_dir, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == 'cuda'))

    start_epoch = 1
    best_val_mse = float('inf')
    best_path = ""

    # optional resume
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_mse = ckpt.get('best_val_mse', best_val_mse)
        print(f"[train] Resumed from {resume_path} at epoch {start_epoch - 1}")

    history = []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = {'mse': 0.0, 'mae': 0.0}
        seen = 0

        print(f"\n[train] Epoch {epoch}/{epochs}")
        for step, (imgs, _) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(amp and device.type == 'cuda')):
                rec, _ = model(imgs)
                mse = F.mse_loss(rec, imgs)
                if loss_mix is None:
                    loss = mse
                else:
                    mae = F.l1_loss(rec, imgs)
                    loss = (1.0 - loss_mix) * mse + loss_mix * mae

            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt)
            scaler.update()

            b = imgs.size(0)
            running['mse'] += mse.detach().item() * b
            if loss_mix is not None:
                running['mae'] += F.l1_loss(rec.detach(), imgs, reduction='mean').item() * b
            else:
                running['mae'] += F.l1_loss(rec.detach(), imgs, reduction='mean').item() * b
            seen += b

            if log_interval and step % log_interval == 0:
                print(f"  step {step:5d} | mse {running['mse'] / seen:.6f} | mae {running['mae'] / seen:.6f}")

        # end epoch — training metrics
        train_metrics = {k: v / max(seen, 1) for k, v in running.items()}

        # validation
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, device, sample_dir=sample_dir, epoch=epoch)
        else:
            val_metrics = {'mse': float('nan'), 'mae': float('nan'), 'psnr': float('nan')}

        print(f"  train: mse={train_metrics['mse']:.6f}  mae={train_metrics['mae']:.6f}")
        print(f"  valid: mse={val_metrics['mse']:.6f}  mae={val_metrics['mae']:.6f}  psnr={val_metrics['psnr']:.2f} dB")

        # save 'latest'
        latest_path = os.path.join(ckpt_dir, 'latest.pt')
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'best_val_mse': best_val_mse, }, latest_path)

        # save best (by val MSE if val exists; otherwise by train MSE)
        score = val_metrics['mse'] if not math.isnan(val_metrics['mse']) else train_metrics['mse']
        if score < best_val_mse:
            best_val_mse = score
            best_path = os.path.join(ckpt_dir, 'best.pt')
            torch.save(
                {'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                 'best_val_mse': best_val_mse, }, best_path)
            print(f"  ↑ new best (mse {best_val_mse:.6f}) -> {best_path}")

        # periodic
        if save_every and (epoch % save_every == 0):
            pth = os.path.join(ckpt_dir, f'ep{epoch:03d}.pt')
            torch.save(
                {'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                 'best_val_mse': best_val_mse, }, pth)
            print(f"  saved {pth}")

        # store history row
        row = {'epoch': epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row['best_ckpt'] = best_path
        history.append(row)

    return {'history': history, 'best_ckpt': best_path, 'best_val_mse': best_val_mse, 'outdir': outdir, }
