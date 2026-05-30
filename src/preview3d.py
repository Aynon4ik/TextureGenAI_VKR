import logging
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SHAPE_CHOICES_RU = {
    "cube": "Куб",
    "sphere": "Сфера",
    "cylinder": "Цилиндр",
}
SHAPE_FROM_RU = {v: k for k, v in SHAPE_CHOICES_RU.items()}
SHAPE_ORDER_RU = tuple(SHAPE_CHOICES_RU.values())
DEFAULT_SHAPE_RU = "Куб"
SHAPE_KEYS = tuple(SHAPE_CHOICES_RU.keys())


def _to_pil_image(img) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        pil = img
    else:
        arr = np.asarray(img)
        if arr.dtype in (np.float32, np.float64) and arr.size and arr.max() <= 1.0 + 1e-3:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            pil = Image.fromarray(arr, mode="L")
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            pil = Image.fromarray(arr)
        else:
            raise TypeError(f"Unsupported image array shape: {getattr(arr, 'shape', None)}")
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGBA")
    return pil


def _mesh_dict(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray) -> dict:
    return {
        "vertices": vertices.astype(np.float32),
        "faces": faces.astype(np.uint32),
        "uvs": uvs.astype(np.float32),
    }


def _create_sphere_mesh(radius: float = 1.0, slices: int = 40, stacks: int = 20) -> dict:
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[list[int]] = []

    def push_vertex(x: float, y: float, z: float, u: float, v: float) -> int:
        vertices.append([x, y, z])
        uvs.append([u, v])
        return len(vertices) - 1

    for j in range(stacks + 1):
        v = j / stacks
        theta = np.pi * v
        y = radius * np.cos(theta)
        ring_r = radius * np.sin(theta)

        for i in range(slices + 1):
            u = i / slices
            phi = 2.0 * np.pi * u
            x = ring_r * np.sin(phi)
            z = ring_r * np.cos(phi)
            push_vertex(x, y, z, u, 1.0 - v)

    row = slices + 1
    for j in range(stacks):
        for i in range(slices):
            a = j * row + i
            b = a + row
            c = a + 1
            d = b + 1
            faces.append([a, b, c])
            faces.append([c, b, d])

    return _mesh_dict(
        np.array(vertices, dtype=np.float64),
        np.array(faces, dtype=np.uint32),
        np.array(uvs, dtype=np.float64),
    )


def _create_cube_mesh(size: float = 2.0) -> dict:
    h = size / 2.0
    vertices = np.array([
        [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],
        [ h, -h,  h], [ h, -h, -h], [ h,  h, -h], [ h,  h,  h],
        [ h, -h, -h], [-h, -h, -h], [-h,  h, -h], [ h,  h, -h],
        [-h, -h, -h], [-h, -h,  h], [-h,  h,  h], [-h,  h, -h],
        [-h,  h,  h], [ h,  h,  h], [ h,  h, -h], [-h,  h, -h],
        [-h, -h, -h], [ h, -h, -h], [ h, -h,  h], [-h, -h,  h],
    ], dtype=np.float64)
    uvs = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float64)
    faces_list = []
    for face in range(6):
        b = face * 4
        faces_list.append([b, b + 1, b + 2])
        faces_list.append([b, b + 2, b + 3])
    faces = np.array(faces_list, dtype=np.uint32)
    return _mesh_dict(vertices, faces, uvs)


def _create_cylinder_mesh(radius: float = 1.0, height: float = 2.0, sections: int = 32) -> dict:
    half = height / 2.0
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[list[int]] = []

    def push_vertex(x: float, y: float, z: float, u: float, v: float) -> int:
        vertices.append([x, y, z])
        uvs.append([u, v])
        return len(vertices) - 1

    for i in range(sections):
        a0 = 2.0 * np.pi * i / sections
        a1 = 2.0 * np.pi * (i + 1) / sections
        u0, u1 = i / sections, (i + 1) / sections

        x0, z0 = radius * np.cos(a0), radius * np.sin(a0)
        x1, z1 = radius * np.cos(a1), radius * np.sin(a1)

        bl = push_vertex(x0, -half, z0, u0, 0.08)
        tl = push_vertex(x0, half, z0, u0, 0.72)
        br = push_vertex(x1, -half, z1, u1, 0.08)
        tr = push_vertex(x1, half, z1, u1, 0.72)

        faces.append([bl, tr, br])
        faces.append([bl, tl, tr])

    center_top = push_vertex(0.0, half, 0.0, 0.5, 0.92)
    top_ring = []
    for i in range(sections):
        a = 2.0 * np.pi * i / sections
        x, z = radius * np.cos(a), radius * np.sin(a)
        top_ring.append(
            push_vertex(x, half, z, 0.5 + 0.48 * np.cos(a), 0.92 + 0.06 * np.sin(a))
        )
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([center_top, top_ring[j], top_ring[i]])

    center_bot = push_vertex(0.0, -half, 0.0, 0.5, 0.02)
    bot_ring = []
    for i in range(sections):
        a = 2.0 * np.pi * i / sections
        x, z = radius * np.cos(a), radius * np.sin(a)
        bot_ring.append(
            push_vertex(x, -half, z, 0.5 + 0.48 * np.cos(a), 0.02 - 0.06 * np.sin(a))
        )
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([center_bot, bot_ring[i], bot_ring[j]])

    return _mesh_dict(
        np.array(vertices, dtype=np.float64),
        np.array(faces, dtype=np.uint32),
        np.array(uvs, dtype=np.float64),
    )


def create_mesh(shape: str = "sphere") -> dict:
    key = shape if shape in SHAPE_KEYS else "cube"
    if key == "cube":
        return _create_cube_mesh()
    if key == "cylinder":
        return _create_cylinder_mesh()
    return _create_sphere_mesh()


def _apply_tile_repeat(uvs: np.ndarray, repeat: int) -> np.ndarray:
    r = max(1, int(repeat))
    if r <= 1:
        return uvs
    return uvs * float(r)


def _prepare_texture_for_glb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        a_max = int(np.array(a, dtype=np.uint8).max())
        rgb = Image.merge("RGB", (r, g, b))
        if a_max < 8:
            img = rgb
        else:
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(rgb, mask=a)
            img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize((1024, 1024), Image.LANCZOS)


def _preview_output_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    d = root / "models" / "preview_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_pbr_material(
    view_mode: str,
    maps: dict | None,
    diffuse: Image.Image | None,
):
    from trimesh.visual.texture import PBRMaterial

    maps = maps or {}
    albedo = maps.get("albedo") or diffuse

    if view_mode == "albedo" and albedo is not None:
        return PBRMaterial(
            baseColorTexture=_prepare_texture_for_glb(albedo),
            metallicFactor=0.0,
            roughnessFactor=0.88,
            doubleSided=True,
        )

    if view_mode in ("normal", "roughness", "ao", "height"):
        show = maps.get(view_mode, albedo)
        if show is None:
            show = albedo
        return PBRMaterial(
            baseColorTexture=_prepare_texture_for_glb(show) if show is not None else None,
            metallicFactor=0.0,
            roughnessFactor=0.92,
            doubleSided=True,
        )

    if albedo is None:
        return PBRMaterial(
            baseColorFactor=[0.72, 0.72, 0.78, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.85,
            doubleSided=True,
        )

    show = maps.get(view_mode, albedo)
    return PBRMaterial(
        baseColorTexture=_prepare_texture_for_glb(show),
        metallicFactor=0.0,
        roughnessFactor=0.85,
        doubleSided=True,
    )


def _export_glb(mesh: dict, path: str, material) -> None:
    import trimesh
    from trimesh.visual.texture import TextureVisuals

    vis = TextureVisuals(uv=mesh["uvs"], material=material)
    tri = trimesh.Trimesh(
        vertices=mesh["vertices"],
        faces=mesh["faces"],
        visual=vis,
        process=False,
    )
    tri.export(path)


def apply_texture(
    diffuse: Image.Image | None = None,
    shape: str = "sphere",
    tile_repeat: int = 1,
    view_mode: str = "albedo",
    pbr_maps: dict | None = None,
) -> str:
    diffuse = _to_pil_image(diffuse)

    mesh = create_mesh(shape)
    mesh = {
        **mesh,
        "uvs": _apply_tile_repeat(mesh["uvs"], tile_repeat),
    }

    mat = _build_pbr_material(view_mode, pbr_maps, diffuse)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".glb", delete=False, dir=str(_preview_output_dir())
    )
    tmp.close()
    _export_glb(mesh, tmp.name, mat)
    logger.info("Textured %s (tile=%s, mode=%s) -> %s", shape, tile_repeat, view_mode, tmp.name)
    return str(Path(tmp.name).resolve())
