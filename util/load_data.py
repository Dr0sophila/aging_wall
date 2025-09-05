import os
from typing import List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def load_images_as_tensor(
    dir_path: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
    mode: str = "RGB",                       # "L" for grayscale, "RGB" for color
    resize: Optional[Union[int, Tuple[int, int]]] = None,  # (H, W) or int for square
    recursive: bool = False,
    normalize: bool = True,               # scale to [0,1]
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    return_paths: bool = False
):
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
    sizes: List[Tuple[int, int]] = []

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
            t = t.unsqueeze(0)      # [1, H, W]

        t = t.to(dtype)
        if normalize:
            t = t / 255.0

        tensors.append(t)
        sizes.append((t.shape[1], t.shape[2]))  # (H, W)

    # Ensure stackable: pad to max H/W if needed
    H_max = max(h for h, _ in sizes)
    W_max = max(w for _, w in sizes)

    padded = []
    for t in tensors:
        _, h, w = t.shape
        # pad format: (pad_left, pad_right, pad_top, pad_bottom)
        pad = (0, W_max - w, 0, H_max - h)
        if pad != (0, 0, 0, 0):
            t = F.pad(t, pad, mode="constant", value=0)
        padded.append(t)

    imgs = torch.stack(padded, dim=0)  # [N, C, H_max, W_max]
    if device is not None:
        imgs = imgs.to(device)

    if return_paths:
        return imgs, files
    return imgs
