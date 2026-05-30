from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def _to_rgb_array(image: Image.Image) -> tuple[np.ndarray, Image.Image | None]:
    if image.mode == "RGBA":
        rgba = np.array(image, dtype=np.float64)
        return rgba[:, :, :3], Image.fromarray(rgba[:, :, 3].astype(np.uint8), mode="L")
    rgb = np.array(image.convert("RGB"), dtype=np.float64)
    return rgb, None


def _from_rgb_array(rgb: np.ndarray, alpha: Image.Image | None) -> Image.Image:
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(alpha)
    return out


def make_seamless(
    image: Image.Image,
    blend_width: int = 64,
    blend_ratio: float | None = None,
) -> Image.Image:
    img, alpha = _to_rgb_array(image)
    h, w = img.shape[:2]

    rolled = np.roll(np.roll(img, h // 2, axis=0), w // 2, axis=1)

    y = np.abs(np.linspace(-1, 1, h))
    x = np.abs(np.linspace(-1, 1, w))
    Y, X = np.meshgrid(y, x, indexing="ij")
    mask = np.maximum(X, Y)

    if blend_ratio is not None:
        ratio = float(blend_ratio)
    else:
        ratio = max(float(blend_width) / 256.0, 0.05)
    sigma = int(min(h, w) * ratio)
    mask = gaussian_filter(mask, sigma=max(sigma, 1))
    lo, hi = mask.min(), mask.max()
    mask = (mask - lo) / (hi - lo + 1e-8)
    mask = mask[:, :, np.newaxis]

    result = mask * rolled + (1.0 - mask) * img
    return _from_rgb_array(result, alpha)


def create_tiling_preview(image: Image.Image, repeats: int = 3) -> Image.Image:
    base = image.convert("RGB")
    w, h = base.size
    preview = Image.new("RGB", (w * repeats, h * repeats))
    for row in range(repeats):
        for col in range(repeats):
            preview.paste(base, (col * w, row * h))
    return preview
