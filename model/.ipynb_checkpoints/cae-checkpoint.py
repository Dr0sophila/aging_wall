import os, math, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# ------------- Blocks (unchanged) -----------------
class DownBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU, with optional downsampling via stride."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UpBlock(nn.Module):
    """Upsample by 2x (ConvT) -> Conv block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

# ------------- VQ (EMA) ---------------------------
class VectorQuantizerEMA(nn.Module):
    """
    Exponential moving average VQ from VQ-VAE.
    Inputs are (B, C, H, W). Codebook is (K, D) with D==C.
    Returns quantized x_q, codebook indices, EMA vq loss, perplexity.
    """
    def __init__(self, n_codes: int, code_dim: int, decay: float = 0.99, eps: float = 1e-5, beta: float = 0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.beta = beta

        embed = torch.randn(n_codes, code_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", embed.clone())

    @torch.no_grad()
    def _ema_update(self, flat_inputs, encodings):
        # encodings: (N, K) one-hot
        cluster_size = encodings.sum(0)  # (K,)
        embed_sum = encodings.t() @ flat_inputs  # (K, D)

        # EMA updates
        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum,   alpha=1 - self.decay)

        # Laplace smoothing to avoid empty codes
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
        self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))

    def forward(self, z_e):
        """
        z_e: (B, C, H, W)
        """
        B, C, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (N, C)

        # Distances to embedding vectors
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x·e
        x2 = (flat ** 2).sum(dim=1, keepdim=True)                # (N, 1)
        e2 = (self.embed ** 2).sum(dim=1, keepdim=True).t()      # (1, K)
        distances = x2 + e2 - 2 * flat @ self.embed.t()          # (N, K)

        # Nearest neighbors
        indices = torch.argmin(distances, dim=1)                 # (N,)
        encodings = F.one_hot(indices, num_classes=self.n_codes).type(flat.dtype)

        # Quantize
        z_q = (encodings @ self.embed).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # EMA codebook update
        if self.training:
            self._ema_update(flat.detach(), encodings.detach())

        # Losses (straight-through)
        # commitment: ||sg[z_e] - z_q||^2 ; codebook (EMA) via moving averages (no explicit codebook loss here)
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        vq_loss = self.beta * commitment_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        # Perplexity
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, indices.view(B, H, W), vq_loss, perplexity

# ------------- VQ-VAE model -----------------------
class VQVAE(nn.Module):
    """
    VQ-VAE with your Encoder/Decoder:
      Encoder: 256 -> 128 -> 64 -> 32 -> 16 (channels: 32, 64, 128, 256)
      Quantizer: EMA codebook on (B, ch4, 16, 16)
      Decoder: mirror back to 256
    """
    def __init__(self, in_channels: int = 3, base_ch: int = 32,
                 n_codes: int = 1024, code_dim: int = 256, vq_decay: float = 0.99, vq_beta: float = 0.25):
        super().__init__()
        ch1 = base_ch          # 32
        ch2 = base_ch * 2      # 64
        ch3 = base_ch * 4      # 128
        ch4 = base_ch * 8      # 256

        # Encoder
        self.enc1 = DownBlock(in_channels, ch1, stride=2)   # 256 -> 128
        self.enc2 = DownBlock(ch1, ch2,        stride=2)    # 128 -> 64
        self.enc3 = DownBlock(ch2, ch3,        stride=2)    # 64  -> 32
        self.enc4 = DownBlock(ch3, ch4,        stride=2)    # 32  -> 16

        # Optional 1x1 conv to match code_dim if different
        self.enc_to_code = nn.Conv2d(ch4, code_dim, kernel_size=1, bias=False) if code_dim != ch4 else nn.Identity()
        self.code_to_dec = nn.Conv2d(code_dim, ch4, kernel_size=1, bias=False) if code_dim != ch4 else nn.Identity()

        # VQ (EMA)
        self.vq = VectorQuantizerEMA(n_codes=n_codes, code_dim=code_dim, decay=vq_decay, beta=vq_beta)

        # Decoder
        self.dec1 = UpBlock(ch4, ch3)   # 16 -> 32
        self.dec2 = UpBlock(ch3, ch2)   # 32 -> 64
        self.dec3 = UpBlock(ch2, ch1)   # 64 -> 128
        self.dec4 = UpBlock(ch1, ch1)   # 128 -> 256

        self.out = nn.Sequential(
            nn.Conv2d(ch1, in_channels, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),  # inputs in [0,1]
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
        x = self.enc4(x)    # [B, ch4, 16, 16]
        return x

    def quantize(self, h):
        z_e = self.enc_to_code(h)             # (B, code_dim, 16, 16)
        z_q, idx, vq_loss, perplexity = self.vq(z_e)
        h_q = self.code_to_dec(z_q)           # (B, ch4, 16, 16)
        return h_q, idx, vq_loss, perplexity

    def decode(self, h_q):
        x = self.dec1(h_q)    # [B, ch3, 32, 32]
        x = self.dec2(x)      # [B, ch2, 64, 64]
        x = self.dec3(x)      # [B, ch1,128,128]
        x = self.dec4(x)      # [B, ch1,256,256]
        return self.out(x)

    def forward(self, x):
        h = self.encode(x)
        h_q, idx, vq_loss, perplexity = self.quantize(h)
        x_rec = self.decode(h_q)
        return x_rec, h_q, idx, vq_loss, perplexity


def _ensure_nchw_t(x: torch.Tensor) -> torch.Tensor:
    """Accept NCHW or NHWC; return NCHW float32 (no copy if already ok)."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")
    if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):  # likely NHWC
        x = x.permute(0, 3, 1, 2).contiguous()
    if x.dtype != torch.float32:
        x = x.float()
    return x

@torch.no_grad()
def _mse_per_sample_t(model, arr_t: torch.Tensor, device, batch_size=256):
    """Compute per-sample MSE over a tensor dataset using the model."""
    model.eval()
    scores = []
    N = arr_t.shape[0]
    for i in range(0, N, batch_size):
        batch = arr_t[i:i+batch_size].to(device, non_blocking=True)
        rec, *_ = model(batch)
        per = F.mse_loss(rec, batch, reduction='none').view(batch.size(0), -1).mean(dim=1)
        scores.extend(per.detach().cpu().tolist())
    return scores

def _roc_auc_from_scores(y_true, y_score):
    """ROC AUC via Mann–Whitney U (no sklearn)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = y_score.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    r_pos = ranks[y_true == 1].sum()
    U = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))

def _build_fixed_mixed_loader_t(
    normal_t: torch.Tensor,
    abnormal_t: torch.Tensor,
    abnormal_rate: float,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
):
    """
    Create the mixed dataset ONCE:
      Nn normals (all) + floor(Nn * abnormal_rate) abnormals (with wrap if needed).
    Return a single DataLoader reused across all epochs (shuffle=True reshuffles order each epoch).
    """
    g = torch.Generator().manual_seed(seed)

    Nn = normal_t.shape[0]
    Na_req = int(math.floor(Nn * max(0.0, abnormal_rate)))
    Na_have = abnormal_t.shape[0]

    # shuffle normals and use ALL of them
    idx_n = torch.randperm(Nn, generator=g)
    nrm_sel = normal_t.index_select(0, idx_n)

    # build abnormals to required count
    if Na_req <= Na_have:
        idx_a = torch.randperm(Na_have, generator=g)[:Na_req]
        abn_sel = abnormal_t.index_select(0, idx_a)
    else:
        reps = int(math.ceil(Na_req / Na_have))
        abn_sel = abnormal_t.repeat(reps, 1, 1, 1)[:Na_req]
        idx_a = torch.randperm(Na_req, generator=g)
        abn_sel = abn_sel.index_select(0, idx_a)

    mix = torch.cat([nrm_sel, abn_sel], dim=0)

    ds = TensorDataset(mix)  # (x,)
    ld = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,          # same dataset; order reshuffles each epoch
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    return ld

# --------- training (tensors) ----------
def training(
    model,
    normal_t: torch.Tensor,
    abnormal_t: torch.Tensor,
    *,
    abnormal_rate: float = 0.25,
    epochs: int = 50,
    lr: float = 2e-4,
    batch_size: int = 256,
    device=None,
    outdir: str = "runs/vqvae_tensor_fixed",
    resume_path: str = "",
    save_optimizer: bool = False,
    best_filename: str = "best.pt",
    lambda_recon: float = 1.0,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    seed: int = 42,
):
    """
    Train VQ-VAE from two **torch.Tensors** (normal, abnormal) using a FIXED mixed train_loader.
    normal_t / abnormal_t can be NCHW or NHWC; they are converted to NCHW float32 once.
    Each epoch: log train losses; evaluate per-class MSE and AUC (score=MSE, abnormal=1).
    """
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = os.path.join(outdir, "ckpts"); os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, best_filename)

    # Ensure NCHW float32 (on CPU for building the loader)
    normal_t   = _ensure_nchw_t(normal_t.cpu())
    abnormal_t = _ensure_nchw_t(abnormal_t.cpu())

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch, best_auc = 1, -1.0
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            if save_optimizer and 'opt' in ckpt and ckpt['opt'] is not None:
                opt.load_state_dict(ckpt['opt'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_auc = ckpt.get('best_auc', best_auc)
        else:
            model.load_state_dict(ckpt)
        print(f"[train] Resumed from {resume_path} at epoch {start_epoch-1}")

    # ---- build the fixed mixed train loader ONCE ----
    train_loader = _build_fixed_mixed_loader_t(
        normal_t, abnormal_t, abnormal_rate,
        batch_size, seed, num_workers, pin_memory, drop_last
    )

    history = []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = {'mse': 0.0, 'mae': 0.0, 'vq': 0.0, 'total': 0.0}
        seen = 0

        print(f"\n[train] Epoch {epoch}/{epochs}  (fixed loader, abnormal_rate={abnormal_rate})")
        for step, (x_batch,) in enumerate(train_loader, 1):
            x = x_batch.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            out = model(x)
            rec = out[0]
            vq_loss = out[3] if len(out) >= 4 and isinstance(out[3], torch.Tensor) else 0.0

            mse = F.mse_loss(rec, x)
            mae = F.l1_loss(rec, x)
            loss = lambda_recon * mse + (vq_loss if isinstance(vq_loss, torch.Tensor) else 0.0)

            loss.backward()
            opt.step()

            b = x.size(0)
            running['mse'] += mse.detach().item() * b
            running['mae'] += mae.detach().item() * b
            running['vq']  += (vq_loss.detach().item() if isinstance(vq_loss, torch.Tensor) else 0.0) * b
            running['total'] += loss.detach().item() * b
            seen += b

        train_metrics = {k: v / max(seen, 1) for k, v in running.items()}

        # ---- evaluation: per-class MSE + AUC ----
        scores_normal   = _mse_per_sample_t(model, normal_t,   device, batch_size=batch_size)
        scores_abnormal = _mse_per_sample_t(model, abnormal_t, device, batch_size=batch_size)

        eval_mse_normal   = float(np.mean(scores_normal))   if len(scores_normal)   else float("nan")
        eval_mse_abnormal = float(np.mean(scores_abnormal)) if len(scores_abnormal) else float("nan")

        labels = [0]*len(scores_normal) + [1]*len(scores_abnormal)
        scores = scores_normal + scores_abnormal
        auc = _roc_auc_from_scores(labels, scores)

        print(f"  train: mse={train_metrics['mse']:.6f}  mae={train_metrics['mae']:.6f}  vq={train_metrics['vq']:.6f}")
        print(f"  eval : mse_normal={eval_mse_normal:.6f}  mse_abnormal={eval_mse_abnormal:.6f}  AUC={auc:.4f}")

        # ---- save best by AUC ----
        improved = (auc > best_auc) if not math.isnan(auc) else False
        if improved:
            best_auc = auc
            save_blob = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_auc': best_auc,
                'config': {
                    'abnormal_rate': abnormal_rate,
                    'lr': lr,
                    'batch_size': batch_size,
                    'lambda_recon': lambda_recon,
                }
            }
            if save_optimizer:
                save_blob['opt'] = opt.state_dict()
            torch.save(save_blob, best_path)
            print(f"  ↑ new best (AUC {best_auc:.4f}) -> {best_path}")

        history.append({
            'epoch': epoch,
            'train_mse': train_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'train_vq':  train_metrics['vq'],
            'eval_mse_normal': eval_mse_normal,
            'eval_mse_abnormal': eval_mse_abnormal,
            'eval_auc': auc,
            'best_ckpt': best_path
        })

    return {
        'history': history,
        'best_ckpt': best_path,
        'best_auc': best_auc,
        'outdir': outdir
    }