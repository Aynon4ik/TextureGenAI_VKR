from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

MAP_KEYS = ("albedo", "normal", "roughness", "ao", "height")
EXPORT_PBR_KEYS = ("normal", "roughness", "ao", "height")

NORMAL_STRENGTH = 4.0
AO_STRENGTH = 1.2
HEIGHT_EDGE_BOOST = 0.12
ROUGHNESS_BASE = 0.74
ROUGHNESS_DETAIL = 0.14
ROUGHNESS_MORTAR = 0.06


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def _normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def _to_uint8(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def _gray01(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(_luminance(rgb), 0.0, 1.0)


def _pad_wrap(img: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad)), mode="wrap")


def _crop_wrap(padded: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return padded
    return padded[pad:-pad, pad:-pad]


def _gaussian_blur_wrap(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.astype(np.float32)
    pad = max(1, int(round(3 * sigma)))
    padded = _pad_wrap(img.astype(np.float32), pad)
    blurred = gaussian_filter(padded, sigma=sigma)
    return _crop_wrap(blurred, pad)


def _sobel_wrap(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = 1
    h = _pad_wrap(img.astype(np.float32), p)
    gx = (h[p:-p, 2:] - h[p:-p, :-2]) * 0.5
    gy = (h[2:, p:-p] - h[:-2, p:-p]) * 0.5
    return gx, gy


def _contrast_stretch(
    gray: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    floor: float = 0.08,
    ceil: float = 0.94,
) -> np.ndarray:
    lo, hi = np.percentile(gray, [lo_pct, hi_pct])
    out = (gray - lo) / (hi - lo + 1e-8)
    out = np.clip(out, 0.0, 1.0)
    return (floor + out * (ceil - floor)).astype(np.float32)


def _height_physical(gray: np.ndarray) -> np.ndarray:
    brick = _contrast_stretch(1.0 - gray)
    gx, gy = _sobel_wrap(brick)
    edges = np.sqrt(gx * gx + gy * gy)
    edges = edges / (edges.max() + 1e-8)
    h = np.clip(brick + HEIGHT_EDGE_BOOST * edges, 0.0, 1.0)
    return _gaussian_blur_wrap(h, sigma=0.4)


def _height_to_normal(height: np.ndarray, strength: float) -> Image.Image:
    gx, gy = _sobel_wrap(height)
    nx = -gx * strength
    ny = gy * strength
    nz = np.ones_like(height, dtype=np.float32)
    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx = nx / length * 0.5 + 0.5
    ny = ny / length * 0.5 + 0.5
    nz = nz / length * 0.5 + 0.5
    normal = np.stack([nx, ny, nz], axis=-1)
    return Image.fromarray((np.clip(normal, 0, 1) * 255).astype(np.uint8))


def generate_albedo_map(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    rgb = np.clip(rgb, 0.05, 0.96)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def generate_height_map(image: Image.Image) -> Image.Image:
    gray = _gray01(image)
    return _to_uint8(_height_physical(gray))


def generate_normal_map(
    image: Image.Image,
    strength: float = NORMAL_STRENGTH,
) -> Image.Image:
    gray = _gray01(image)
    return _height_to_normal(_height_physical(gray), strength)


def generate_roughness_map(image: Image.Image) -> Image.Image:
    g = _gray01(image)
    c1 = np.abs(g - _gaussian_blur_wrap(g, sigma=1.2))
    c2 = np.abs(g - _gaussian_blur_wrap(g, sigma=2.8))
    detail = _normalize01(0.55 * c1 + 0.45 * c2)
    mortar = _normalize01(_gaussian_blur_wrap(g, sigma=1.8))
    rough = (
        ROUGHNESS_BASE
        + ROUGHNESS_DETAIL * detail
        + ROUGHNESS_MORTAR * mortar
    )
    rough = gaussian_filter(rough, sigma=0.6)
    rough = np.clip(rough, 0.68, 0.94)
    return _to_uint8(rough)


def generate_ao_map(image: Image.Image) -> Image.Image:
    gray = _gray01(image)
    h = _height_physical(gray)
    b2 = _gaussian_blur_wrap(h, sigma=2.0)
    b4 = _gaussian_blur_wrap(h, sigma=4.0)
    b8 = _gaussian_blur_wrap(h, sigma=8.0)
    occl = np.maximum(
        0.0,
        (b2 - h) * 0.5 + (b4 - h) * 0.35 + (b8 - h) * 0.15,
    )
    occl = _normalize01(occl)
    ao = np.clip(1.0 - occl * AO_STRENGTH, 0.0, 1.0)
    return _to_uint8(ao)


def generate_all_maps(image: Image.Image) -> dict[str, Image.Image]:
    albedo = generate_albedo_map(image)
    gray = _gray01(albedo)
    height_f = _height_physical(gray)
    return {
        "albedo": albedo,
        "height": _to_uint8(height_f),
        "normal": _height_to_normal(height_f, NORMAL_STRENGTH),
        "roughness": generate_roughness_map(albedo),
        "ao": generate_ao_map(albedo),
    }
