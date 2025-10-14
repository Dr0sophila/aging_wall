import os
from typing import List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_images_as_tensor(dir_path: str,
                          extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
                          mode: str = "RGB", resize: Optional[Union[int, Tuple[int, int]]] = None,
                          # (H, W) or int for square
                          recursive: bool = False, normalize: bool = True,  # scale to [0,1]
                          dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None):
    """
    Load all images from dir_path into a stacked tensor [N, C, H, W].
    If 'resize' is None and images have different sizes, they are zero-padded to the max H/W.

    Returns:
        imgs: torch.Tensor [N, C, H, W]
        (optional) paths: List[str] in the same order as imgs
    """

    # Collect files
    def is_img(f: str) -> bool:
        return os.path.splitext(f)[1].lower() in extensions

    if recursive:
        files = []
        for root, _, names in os.walk(dir_path):
            for n in names:
                if is_img(n):
                    files.append(os.path.join(root, n))
    else:
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if is_img(f)]

    files.sort()
    if len(files) == 0:
        raise ValueError(f"No images with {extensions} found in '{dir_path}'")

    # Normalize resize argument
    if isinstance(resize, int):
        resize = (resize, resize)  # H, W

    tensors: List[torch.Tensor] = []

    for fp in files:
        with Image.open(fp) as im:
            if mode.upper() not in ("L", "RGB"):
                raise ValueError("mode must be 'L' or 'RGB'")
            im = im.convert(mode.upper())

            if resize is not None:
                # PIL expects (W, H)
                im = im.resize((resize[1], resize[0]), resample=Image.BILINEAR)

            arr = np.array(im)  # H x W (L) or H x W x 3 (RGB)

        t = torch.from_numpy(arr)
        if mode.upper() == "RGB":
            # HWC -> CHW
            t = t.permute(2, 0, 1)  # [3, H, W]
        else:
            # add channel dim
            t = t.unsqueeze(0)  # [1, H, W]

        t = t.to(dtype)
        if normalize:
            t = t / 255.0

        tensors.append(t)

    imgs = torch.stack(tensors, dim=0)  # [N, C, H_max, W_max]
    if device is not None:
        imgs = imgs.to(device)

    return imgs


def load_data(
    normal_path: str,
    abnormal_path: str,
    batch_size: int = 256,
    normal_num: int = 1324,
    abnormal_num: int = 50,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Returns:
        train_ld, val_ld, normal_ld, abnormal_ld  (all DataLoaders)

    - train_ld / val_ld are built from a mixed (normal+abnormal) dataset with labels
    - normal_ld / abnormal_ld are built from their respective subsets (useful for eval)
    """
    # ---------- load ----------
    normal = load_images_as_tensor(normal_path)          # [Nn, C, H, W]
    abnormal = load_images_as_tensor(abnormal_path)      # [Na, C, H, W]

    if normal.dim() != 4 or abnormal.dim() != 4:
        raise ValueError("Expected image tensors with shape [N, C, H, W].")

    # ---------- sample without replacement ----------
    Nn_total = normal.shape[0]
    Na_total = abnormal.shape[0]
    Nn = min(normal_num, Nn_total)
    Na = min(abnormal_num, Na_total)

    g = torch.Generator().manual_seed(seed)
    idx_n = torch.randperm(Nn_total, generator=g)[:Nn]
    idx_a = torch.randperm(Na_total, generator=g)[:Na]

    normal_sel = normal[idx_n].contiguous()      # [Nn, C, H, W]
    abnormal_sel = abnormal[idx_a].contiguous()  # [Na, C, H, W]

    print("normal (sampled) :", tuple(normal_sel.shape))
    print("abnormal(samples):", tuple(abnormal_sel.shape))

    # ---------- build labeled full dataset ----------
    y_n = torch.zeros(Nn, dtype=torch.long)  # label 0 = normal
    y_a = torch.ones(Na, dtype=torch.long)   # label 1 = abnormal

    full_x = torch.cat([normal_sel, abnormal_sel], dim=0)
    full_y = torch.cat([y_n, y_a], dim=0)

    print("full (x,y)      :", tuple(full_x.shape), tuple(full_y.shape))

    full_ds = TensorDataset(full_x, full_y)

    # ---------- split 95/5 ----------
    val_sz = max(1, int(0.05 * len(full_ds)))
    train_sz = len(full_ds) - val_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz], generator=g)

    # ---------- per-class datasets (optional but handy) ----------
    normal_ds = TensorDataset(normal_sel, y_n)
    abnormal_ds = TensorDataset(abnormal_sel, y_a)

    # ---------- loaders ----------
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    normal_ld   = DataLoader(normal_ds,   batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    abnormal_ld = DataLoader(abnormal_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_ld, val_ld, normal_ld, abnormal_ld