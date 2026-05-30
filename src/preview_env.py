from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ASSETS_HDR = Path(__file__).resolve().parent.parent / "assets" / "hdr"

HDR_LABELS = ("Студия", "Улица")
HDR_FROM_RU = {"Студия": "studio", "Улица": "outdoor"}


def _write_gradient_png(
    path: Path,
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
) -> None:
    h, w = 256, 512
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    top_c = np.array(top, dtype=np.float32)
    bot_c = np.array(bottom, dtype=np.float32)
    rgb = (bot_c + (top_c - bot_c) * (1.0 - t))[:, None, :]
    rgb = np.broadcast_to(rgb, (h, w, 3)).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


def ensure_hdr_assets() -> dict[str, Path]:
    ASSETS_HDR.mkdir(parents=True, exist_ok=True)
    paths = {
        "studio": ASSETS_HDR / "studio.png",
        "outdoor": ASSETS_HDR / "outdoor.png",
    }
    if not paths["studio"].exists():
        _write_gradient_png(paths["studio"], top=(95, 92, 88), bottom=(18, 19, 24))
    if not paths["outdoor"].exists():
        _write_gradient_png(paths["outdoor"], top=(142, 197, 240), bottom=(92, 107, 82))
    return paths


HDR_EXR = {
    "studio": ASSETS_HDR / "studio.exr",
    "outdoor": ASSETS_HDR / "outdoor.exr",
}


def hdr_env_status() -> str | None:
    missing = []
    small = []
    for key, path in HDR_EXR.items():
        if not path.exists():
            missing.append(path.name)
        elif path.stat().st_size < 50_000:
            small.append(path.name)
    if missing:
        return f"Нет файлов: {', '.join(missing)} в {ASSETS_HDR}"
    if small:
        return f"Файлы слишком маленькие (битые?): {', '.join(small)}"
    return None
