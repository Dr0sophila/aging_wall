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
class AutoEncoder(nn.Module):
    """
    Fully convolutional 3-stage AE for 256x256 using ONLY DownBlock/UpBlock.
      Encoder: 256 -> 128 -> 64 -> 32   (DownBlock x3)
      Bottleneck: (ch3, 32, 32)
      Decoder: 32 -> 64 -> 128 -> 256   (UpBlock x3)
    """
    def __init__(self, in_channels: int = 3, base_ch: int = 32):
        super().__init__()
        ch1 = base_ch          # 32
        ch2 = base_ch * 2      # 64
        ch3 = base_ch * 4      # 128

        # Encoder
        self.enc1 = DownBlock(in_channels, ch1, stride=2)   # 256 -> 128
        self.enc2 = DownBlock(ch1, ch2,        stride=2)    # 128 -> 64
        self.enc3 = DownBlock(ch2, ch3,        stride=2)    # 64  -> 32

        # Decoder (mirror)
        self.dec1 = UpBlock(ch3, ch2)   # 32 -> 64
        self.dec2 = UpBlock(ch2, ch1)   # 64 -> 128
        self.dec3 = UpBlock(ch1, ch1)   # 128 -> 256

        # Output head (inputs expected in [0,1])
        self.out = nn.Sequential(
            nn.Conv2d(ch1, in_channels, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias  is not None: nn.init.zeros_(m.bias)

    def encode(self, x):
        x = self.enc1(x)    # [B, ch1,128,128]
        x = self.enc2(x)    # [B, ch2, 64, 64]
        x = self.enc3(x)    # [B, ch3, 32, 32]
        return x            # bottleneck feature

    def decode(self, h):
        x = self.dec1(h)    # [B, ch2, 64, 64]
        x = self.dec2(x)    # [B, ch1,128,128]
        x = self.dec3(x)    # [B, ch1,256,256]
        return self.out(x)

    def forward(self, x):
        h = self.encode(x)
        x_rec = self.decode(h)
        return x_rec, h


    
import os, math, torch
import torch.nn.functional as F
from torchvision import utils as vutils

def _get_imgs_from_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch

@torch.no_grad()
def _psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y, reduction='mean').clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)

@torch.no_grad()
def _evaluate(model, loader, device, sample_dir=None, epoch=0, max_grids=2, grid_n=4):
    model.eval()
    totals = {'mse': 0.0, 'mae': 0.0, 'psnr': 0.0}
    count, saved = 0, 0
    for batch in loader:
        imgs = _get_imgs_from_batch(batch).to(device, non_blocking=True)
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
    if count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / count for k, v in totals.items()}

def training(
    model, train_loader, val_loader=None, *,
    epochs=50, lr=2e-4,
    device=None,
    outdir="runs/ae1024", resume_path="", log_interval=50,
    loss_mix=None,
    save_optimizer=False,          # False -> save weights-only (smaller .pt)
    best_filename="best.pt",       # customize if you want
):
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = os.path.join(outdir, "ckpts"); os.makedirs(ckpt_dir, exist_ok=True)
    sample_dir = os.path.join(outdir, "samples"); os.makedirs(sample_dir, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch, best_val_mse = 1, float('inf')
    best_path = os.path.join(ckpt_dir, best_filename)

    # Optional resume (supports either weights-only or full dict)
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            if save_optimizer and 'opt' in ckpt:
                opt.load_state_dict(ckpt['opt'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_mse = ckpt.get('best_val_mse', best_val_mse)
        else:
            model.load_state_dict(ckpt)  # weights-only
        print(f"[train] Resumed from {resume_path} at epoch {start_epoch-1}")

    history = []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = {'mse': 0.0, 'mae': 0.0}
        seen = 0

        print(f"\n[train] Epoch {epoch}/{epochs}")
        for step, batch in enumerate(train_loader, 1):
            imgs = _get_imgs_from_batch(batch).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            rec, _ = model(imgs)
            mse = F.mse_loss(rec, imgs)
            if loss_mix is None:
                loss = mse
                mae_for_log = F.l1_loss(rec, imgs, reduction='mean')
            else:
                mae = F.l1_loss(rec, imgs)
                mae_for_log = mae
                loss = (1.0 - loss_mix) * mse + loss_mix * mae

            loss.backward()
            opt.step()

            b = imgs.size(0)
            running['mse'] += mse.detach().item() * b
            running['mae'] += mae_for_log.detach().item() * b
            seen += b

            if log_interval and step % log_interval == 0:
                print(f"  step {step:5d} | mse {running['mse']/seen:.6f} | mae {running['mae']/seen:.6f}")

        train_metrics = {k: v / max(seen, 1) for k, v in running.items()}

        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, device, sample_dir=sample_dir, epoch=epoch)
        else:
            val_metrics = {'mse': float('nan'), 'mae': float('nan'), 'psnr': float('nan')}

        print(f"  train: mse={train_metrics['mse']:.6f}  mae={train_metrics['mae']:.6f}")
        print(f"  valid: mse={val_metrics['mse']:.6f}  mae={val_metrics['mae']:.6f}  psnr={val_metrics['psnr']:.2f} dB")

        # ---- Only save when improved ----
        score = val_metrics['mse'] if not math.isnan(val_metrics['mse']) else train_metrics['mse']
        if score < best_val_mse:
            best_val_mse = score
            if save_optimizer:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'best_val_mse': best_val_mse,
                }, best_path)
            else:
                # weights-only (smallest file)
                torch.save(model.state_dict(), best_path)
            print(f"  ↑ new best (mse {best_val_mse:.6f}) -> {best_path}")

        row = {'epoch': epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row['best_ckpt'] = best_path
        history.append(row)

    return {'history': history, 'best_ckpt': best_path, 'best_val_mse': best_val_mse, 'outdir': outdir}



def load_best_model(ckpt_path, in_channels=3, base_ch=32, latent_dim=1024, device=None):
    """
    Load a trained AutoEncoder1024 from a best.pt checkpoint.
    Returns: model (on device), checkpoint dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # import your model class
    model = AutoEncoder1024(in_channels=in_channels, base_ch=base_ch, latent_dim=latent_dim)
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model#, ckpt
