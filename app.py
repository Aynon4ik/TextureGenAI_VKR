import io
import logging
import socket
import zipfile
from pathlib import Path

import gradio as gr
from PIL import Image

from src.generator import TextureGenerator, denoise_steps_to_inference
from src.classifier import TextureClassifier
from src.materials import CLASSIFIER_CLASSES, MATERIAL_LABELS_RU, material_dropdown_choices
from src.seamless import make_seamless, create_tiling_preview
from src.agent import TextureAgent
from src.preview3d import (
    apply_texture,
    DEFAULT_SHAPE_RU,
    SHAPE_FROM_RU,
    SHAPE_ORDER_RU,
)
from src.pbr import generate_all_maps
from src.preview_env import HDR_EXR, HDR_LABELS, ensure_hdr_assets, hdr_env_status
from src.preview_viewer import (
    HDR_SWITCH_JS,
    PBR_VIEWER_HEAD_JS,
    gradio_file_url,
    viewer_iframe_html,
    viewer_loading_html,
    viewer_placeholder_html,
)

PREVIEW_SHOW_DOWNLOADS_JS = """() => {
  const el = document.getElementById("preview-actions-wrap");
  if (el) el.classList.remove("preview-actions--hidden");
}"""

PREVIEW_HIDE_DOWNLOADS_JS = """() => {
  const el = document.getElementById("preview-actions-wrap");
  if (el) el.classList.add("preview-actions--hidden");
}"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

_generator: TextureGenerator | None = None
_classifier: TextureClassifier | None = None


def _get_generator() -> TextureGenerator:
    global _generator
    if _generator is None:
        _generator = TextureGenerator()
        _generator.load_model()
    return _generator


def _get_classifier() -> TextureClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TextureClassifier(model_path="models/texture_classifier.pth")
    return _classifier


def _resolve_material_type(
    material_label: str,
    ref_image: Image.Image | None = None,
) -> str:
    key = _LABEL_TO_KEY.get(material_label, "default")
    if key != "default":
        return key
    if ref_image is None:
        gr.Warning(
            "Режим «Авто» без референса: используется общий промпт. "
            "Укажите тип материала вручную."
        )
        return "default"
    clf = _get_classifier()
    material, conf, confident, _ = clf.predict_for_auto(ref_image)
    if not confident:
        auto_ru = ", ".join(MATERIAL_LABELS_RU[c] for c in CLASSIFIER_CLASSES)
        gr.Warning(
            f"Классификатор не уверен ({conf:.0%}) — общий промпт. "
            f"Для точного результата выберите материал вручную. "
            f"Авто: {auto_ru}."
        )
    return material


GEN_GALLERY_COLUMNS = 2


def fn_generate(
    ref_image: Image.Image | None,
    material_label: str,
    strength: float,
    guidance: float,
    steps: int,
    num_images: int,
    progress=gr.Progress(),
):
    if ref_image is None:
        gr.Warning("Загрузите референсное изображение!")
        return []

    material_type = _resolve_material_type(material_label, ref_image)
    denoise_steps = int(steps)
    n_steps = denoise_steps_to_inference(denoise_steps, float(strength))

    progress(0.03, desc="Загрузка модели…")
    gen = _get_generator()

    def on_step(frac: float, desc: str) -> None:
        progress(frac, desc=desc)

    progress(0.0, desc="Генерация текстур…")
    images = gen.generate_from_reference(
        reference_image=ref_image,
        material_type=material_type,
        custom_prompt="",
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=n_steps,
        num_images=int(num_images),
        seed=-1,
        on_progress=on_step,
    )
    progress(1.0, desc="Готово!")
    return images


def fn_generate_text(
    prompt: str,
    guidance: float,
    steps: int,
    num_images: int,
    progress=gr.Progress(),
):
    if not prompt.strip():
        gr.Warning("Введите описание текстуры!")
        return []

    n_steps = int(steps)

    progress(0.03, desc="Загрузка модели…")
    gen = _get_generator()
    if gen.txt2img_pipe is None:
        gen.load_txt2img()

    def on_step(frac: float, desc: str) -> None:
        progress(frac, desc=desc)

    progress(0.0, desc="Генерация текстур…")
    images = gen.generate_from_text(
        prompt=prompt,
        material_type="default",
        guidance_scale=guidance,
        num_inference_steps=n_steps,
        num_images=int(num_images),
        seed=-1,
        on_progress=on_step,
    )
    progress(1.0, desc="Готово!")
    return images


DEFAULT_SEAMLESS_BLEND = 64


SEAMLESS_OFF = "Выкл"
SEAMLESS_ON = "Вкл"


def _preview_seamless_choice(active: bool) -> dict:
    return gr.update(value=SEAMLESS_ON if active else SEAMLESS_OFF)


def _preview_work_texture(
    source: Image.Image | None,
    seamless_active: bool,
) -> Image.Image | None:
    if source is None:
        return None
    if seamless_active:
        return make_seamless(source, blend_width=DEFAULT_SEAMLESS_BLEND)
    return source


def _preview_refresh_3d(
    diffuse_image: Image.Image | None,
    shape_label: str,
    tile_repeat: int,
    mode_label: str,
    pbr_maps: dict | None,
    hdr_label: str,
    progress=None,
):
    if diffuse_image is None or not pbr_maps:
        return (
            None,
            _preview_actions_visible(False),
            fn_tile_dl_button_update(tile_repeat),
            gr.update(),
        )
    if progress:
        progress(0.4, desc="Обновление 3D…")
    maps = generate_all_maps(diffuse_image)
    glb_path = _preview_glb(
        diffuse_image, shape_label, tile_repeat, mode_label, maps
    )
    glb_url = gradio_file_url(glb_path)
    if progress:
        progress(1.0, desc="Готово!")
    return (
        maps,
        _preview_actions_visible(True),
        fn_tile_dl_button_update(tile_repeat),
        gr.update(value=viewer_iframe_html(hdr_label, glb_url=glb_url)),
    )


def fn_preview_diffuse_changed(image: Image.Image | None):
    return False, _preview_seamless_choice(False)


def fn_preview_seamless_changed(
    seamless_choice: str,
    source_image: Image.Image | None,
    shape_label: str,
    tile_repeat: int,
    mode_label: str,
    pbr_maps: dict | None,
    hdr_label: str,
):
    if source_image is None:
        gr.Warning("Загрузите текстуру!")
        return (
            False,
            _preview_seamless_choice(False),
            None,
            _preview_actions_visible(False),
            fn_tile_dl_button_update(tile_repeat),
            gr.update(),
        )

    active = seamless_choice == SEAMLESS_ON
    work = _preview_work_texture(source_image, active)

    refresh = _preview_refresh_3d(
        work, shape_label, tile_repeat, mode_label, pbr_maps, hdr_label
    )
    if refresh[0] is None:
        return (
            active,
            _preview_seamless_choice(active),
            None,
            _preview_actions_visible(False),
            fn_tile_dl_button_update(tile_repeat),
            gr.update(),
        )
    maps, actions, tile_dl, viewer = refresh
    return (
        active,
        _preview_seamless_choice(active),
        maps,
        actions,
        tile_dl,
        viewer,
    )


def fn_generate_unified(
    mode: str,
    ref_image: Image.Image | None,
    prompt: str,
    ref_material_label: str,
    strength: float,
    ref_guidance: float,
    ref_steps: int,
    ref_num_images: int,
    txt_guidance: float,
    txt_steps: int,
    txt_num_images: int,
    progress=gr.Progress(),
):
    if mode == "По референсу":
        return fn_generate(
            ref_image,
            ref_material_label,
            strength,
            ref_guidance,
            ref_steps,
            ref_num_images,
            progress,
        )
    return fn_generate_text(
        prompt,
        txt_guidance,
        txt_steps,
        txt_num_images,
        progress,
    )


def _toggle_gen_mode(mode: str):
    is_ref = mode == "По референсу"
    return (
        gr.update(visible=is_ref),
        gr.update(visible=not is_ref),
        gr.update(elem_classes=[] if is_ref else ["gen-panel-hidden"]),
        gr.update(elem_classes=["gen-panel-hidden"] if is_ref else []),
    )


def fn_tile_dl_button_update(tile_repeat: float | int) -> dict:
    show = int(float(tile_repeat)) >= 2
    return gr.update(elem_classes=[] if show else ["tile-dl-hidden"])


def _preview_actions_visible(show: bool) -> dict:
    if show:
        return gr.update(elem_classes=["preview-actions"])
    return gr.update(elem_classes=["preview-actions", "preview-actions--hidden"])


STRATEGY_NAMES = {
    "conservative": "Консервативная",
    "balanced": "Сбалансированная",
    "creative": "Креативная",
    "high-guidance": "Высокая точность",
    "low-guidance": "Свободная генерация",
}


def _format_agent_report(result) -> str:
    mat = result.material_type
    conf = result.material_confidence
    best_strategy = STRATEGY_NAMES.get(
        result.best_params.get("strategy", ""), result.best_params.get("strategy", "?")
    )
    score_pct = result.best_score * 100

    mat_display = MATERIAL_LABELS.get(mat, mat)
    if conf >= 1.0:
        mat_line = f"Материал (вручную): {mat_display}"
    elif not result.material_confident:
        mat_line = (
            f"Материал: общий промпт (классификатор не уверен, {conf:.0%}). "
            "Для бумаги, бетона и др. выберите тип вручную."
        )
    else:
        mat_line = f"Определён материал: {mat_display} (уверенность {conf:.0%})"

    lines = [
        mat_line,
        f"Проверено стратегий: {result.iterations_used}",
        f"Лучшая стратегия: {best_strategy}",
        f"Итоговое качество: {score_pct:.1f} / 100",
        "",
        "Результаты по стратегиям:",
    ]

    for a in result.all_attempts:
        m = a["metrics"]
        s_name = STRATEGY_NAMES.get(a["strategy"], a["strategy"])
        s_pct = a["score"] * 100
        lines.append(
            f"  {a['iteration']}. {s_name} — {s_pct:.1f} баллов "
            f"(цвет {m['color_match']:.0%}, "
            f"детали {m['complexity']:.0%}, "
            f"бесшовность {m['seamless']:.0%}, "
            f"сходство {m['ssim']:.0%})"
        )

    return "\n".join(lines)


def fn_agent_run(
    ref_image: Image.Image | None,
    material_label: str,
    max_iters: int,
    quality_thresh: float,
    progress=gr.Progress(),
):
    if ref_image is None:
        gr.Warning("Загрузите референсное изображение!")
        yield None, [], ""
        return

    material_override = _LABEL_TO_KEY.get(material_label, "default")
    n_iters = max(1, int(max_iters))

    try:
        progress(0.02, desc="Загрузка моделей…")
        gen = _get_generator()
        clf = _get_classifier()

        agent = TextureAgent(
            generator=gen,
            classifier=clf,
            quality_threshold=float(quality_thresh),
            max_iterations=n_iters,
        )

        yield None, [], "Подготовка…"

        for i, result in enumerate(
            agent.iter_run(
                reference_image=ref_image,
                custom_prompt="",
                material_override=material_override,
            )
        ):
            frac = (i + 1) / n_iters
            last_strategy = (
                result.all_attempts[-1]["strategy"]
                if result.all_attempts
                else "…"
            )
            progress(
                frac,
                desc=f"Итерация {result.iterations_used}/{n_iters} ({last_strategy})…",
            )
            gallery = [a["image"] for a in result.all_attempts]
            report = _format_agent_report(result)
            yield result.best_image, gallery, report

        progress(1.0, desc="Готово!")
    except Exception as exc:
        logger.exception("ML-Agent failed")
        gr.Warning(f"Ошибка агента: {exc}")
        yield None, [], f"Ошибка агента:\n{exc}"


ensure_hdr_assets()
_hdr_err = hdr_env_status()
if _hdr_err:
    logger.warning("HDR: %s", _hdr_err)
else:
    logger.info(
        "HDR EXR: studio %.1f MB, outdoor %.1f MB",
        HDR_EXR["studio"].stat().st_size / 1_048_576,
        HDR_EXR["outdoor"].stat().st_size / 1_048_576,
    )
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

_PREVIEW_EXPORT_DIR = (
    Path(__file__).resolve().parent / "models" / "preview_cache" / "exports"
)


def _export_tiling_png(diffuse: Image.Image, tile_repeat: int) -> str:
    repeat = max(1, int(tile_repeat))
    tile = create_tiling_preview(diffuse, repeats=min(repeat, 6))
    _PREVIEW_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _PREVIEW_EXPORT_DIR / "texture_tiling.png"
    tile.save(path, format="PNG")
    return str(path.resolve())


PBR_MODE_RU = ["Albedo", "Normal", "Roughness", "AO", "Height"]
PBR_MODE_FROM_RU = {
    "Albedo": "albedo",
    "Normal": "normal",
    "Roughness": "roughness",
    "AO": "ao",
    "Height": "height",
}


def _preview_glb(
    diffuse_image: Image.Image,
    shape_label: str,
    tile_repeat: int,
    mode_label: str,
    pbr_maps: dict,
) -> str:
    shape = SHAPE_FROM_RU.get(shape_label, "cube")
    repeat = max(1, int(tile_repeat))
    mode = PBR_MODE_FROM_RU.get(mode_label, "albedo")
    return apply_texture(
        diffuse=diffuse_image,
        shape=shape,
        tile_repeat=repeat,
        view_mode=mode,
        pbr_maps=pbr_maps,
    )


def fn_preview3d(
    diffuse_image: Image.Image | None,
    shape_label: str,
    tile_repeat: int,
    mode_label: str,
    hdr_label: str,
    seamless_active: bool,
):
    if diffuse_image is None:
        gr.Warning("Загрузите текстуру!")
        return (
            None,
            _preview_actions_visible(False),
            fn_tile_dl_button_update(1),
            gr.update(),
        )

    work = _preview_work_texture(diffuse_image, seamless_active)
    tile_dl = fn_tile_dl_button_update(tile_repeat)
    hidden_actions = _preview_actions_visible(False)

    yield (
        gr.update(),
        hidden_actions,
        tile_dl,
        gr.update(value=viewer_loading_html("Генерация PBR-карт…", 25)),
    )

    maps = generate_all_maps(work)

    yield (
        gr.update(),
        hidden_actions,
        tile_dl,
        gr.update(value=viewer_loading_html("3D-превью…", 55)),
    )

    glb_path = _preview_glb(work, shape_label, tile_repeat, mode_label, maps)
    glb_url = gradio_file_url(glb_path)

    yield (
        maps,
        _preview_actions_visible(True),
        tile_dl,
        gr.update(value=viewer_iframe_html(hdr_label, glb_url=glb_url)),
    )


def fn_preview_update(
    diffuse_image: Image.Image | None,
    shape_label: str,
    tile_repeat: int,
    mode_label: str,
    pbr_maps: dict | None,
    hdr_label: str,
    seamless_active: bool,
):
    if diffuse_image is None or not pbr_maps:
        return "", gr.update()
    work = _preview_work_texture(diffuse_image, seamless_active)
    glb_path = _preview_glb(work, shape_label, tile_repeat, mode_label, pbr_maps)
    glb_url = gradio_file_url(glb_path)
    return glb_url, gr.update(value=viewer_iframe_html(hdr_label, glb_url=glb_url))


def _export_pbr_png(maps: dict, mode_label: str) -> str:
    mode = PBR_MODE_FROM_RU.get(mode_label, "albedo")
    _PREVIEW_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _PREVIEW_EXPORT_DIR / f"texture_{mode}.png"
    maps[mode].save(path, format="PNG")
    return str(path.resolve())


def _export_pbr_zip(maps: dict) -> str:
    from src.pbr import EXPORT_PBR_KEYS

    _PREVIEW_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _PREVIEW_EXPORT_DIR / "texture_pbr_all.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in EXPORT_PBR_KEYS:
            img = maps.get(key)
            if img is None:
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            zf.writestr(f"texture_{key}.png", buf.getvalue())
    return str(zip_path.resolve())


def fn_dl_current_pbr(pbr_maps: dict | None, mode_label: str):
    if not pbr_maps:
        return None
    return _export_pbr_png(pbr_maps, mode_label)


def fn_dl_all_pbr(pbr_maps: dict | None):
    if not pbr_maps:
        return None
    return _export_pbr_zip(pbr_maps)


def fn_dl_tiling(
    diffuse_image: Image.Image | None,
    tile_repeat: int,
    pbr_maps: dict | None,
    seamless_active: bool,
):
    if diffuse_image is None or not pbr_maps:
        return None
    work = _preview_work_texture(diffuse_image, seamless_active)
    return _export_tiling_png(work, tile_repeat)


MATERIAL_LABELS = dict(MATERIAL_LABELS_RU)
MATERIAL_CHOICES = material_dropdown_choices()
_LABEL_TO_KEY = {v: k for k, v in MATERIAL_LABELS.items()}

CUSTOM_CSS = """
/* ── Title ───────────────────────────────────────── */
.title-row {
    overflow-x: hidden !important;
    max-width: 100% !important;
}
.title-row h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    letter-spacing: -0.5px;
    overflow: hidden !important;
    max-width: 100% !important;
}
.gradio-container {
    overflow-x: hidden !important;
}

/* ── Footer & share buttons ──────────────────────── */
footer { display: none !important; }
button[title="Share"], .share-btn, [aria-label="Share"],
button[title="share"], [aria-label="share"],
.icon-button-wrapper:last-child,
.download-container button:not(:first-child) {
    display: none !important;
}

/* ── Подписи поверх изображений и галерей ─────────── */
#ref-image-upload label,
#gen-results-gallery label,
#agent-attempts-gallery label,
.gradio-container .image-container label.float,
.gradio-container .gallery label.float {
    background: #ffffff !important;
    color: #334155 !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12) !important;
    border: 1px solid #e2e8f0 !important;
    opacity: 1 !important;
}
#ref-image-upload label span,
#gen-results-gallery label span,
#agent-attempts-gallery label span,
.gradio-container .image-container label.float span,
.gradio-container .gallery label.float span {
    opacity: 1 !important;
}

/* ── Hide fullscreen button ──────────────────────── */
button[aria-label="View in full screen"],
button[title="Full screen"],
.icon-button-wrapper button[aria-label*="full" i],
.image-toolbar button:has(svg path[d*="M4.5 11"]) {
    display: none !important;
}

/* ── Left column ─────────────────────────────────── */
.fixed-left { flex-shrink: 0 !important; }

/* Референс: без рамки (пусто = dashed, с фото = solid block) */
.gradio-container-6-10-0 #ref-image-wrap #ref-image-upload,
.gradio-container-6-10-0 #ref-image-wrap #ref-image-upload.block,
.gradio-container-6-10-0 #ref-image-wrap .placeholder,
.gradio-container-6-10-0 #ref-image-wrap [class*="placeholder"],
.gradio-container-6-10-0 #ref-image-upload,
.gradio-container-6-10-0 #ref-image-upload.block {
    --block-border-width: 0 !important;
    border: none !important;
    border-width: 0 !important;
    border-style: none !important;
    outline: none !important;
    box-shadow: none !important;
}
.gradio-container-6-10-0 #ref-image-wrap button.svelte-1o7nwih,
.gradio-container-6-10-0 #ref-image-wrap .image-container,
.gradio-container-6-10-0 #ref-image-wrap .image-frame {
    border: none !important;
    border-width: 0 !important;
    border-style: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Режимы генерации: панели параметров скрывать через CSS (visible=False ломает ползунки) */
#ref-gen-controls.gen-panel-hidden,
#txt-gen-controls.gen-panel-hidden {
    display: none !important;
}

/* Кнопки скачивания */
#preview-actions-wrap.preview-actions--hidden,
.preview-actions.preview-actions--hidden {
    display: none !important;
}
#preview-actions-wrap {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}
.preview-actions-block {
    gap: 6px !important;
    width: 100% !important;
}
.preview-actions-block > .form,
.preview-actions-block > .column {
    gap: 6px !important;
}
.preview-actions {
    gap: 8px !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    width: 100% !important;
}
.preview-actions .block,
.preview-actions > .form,
.preview-actions-wrap .block {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    overflow: visible !important;
}
.preview-actions > button,
.preview-actions > a,
.preview-actions label {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
.tile-dl-hidden {
    display: none !important;
}

/* Галереи-вывод: без загрузки и без ложного скролла */
#gen-results-gallery .upload,
#gen-results-gallery [data-testid="upload"],
#agent-attempts-gallery .upload,
#agent-attempts-gallery [data-testid="upload"] {
    display: none !important;
}
#gen-results-gallery,
#gen-results-gallery > .block,
#gen-results-gallery .grid-wrap,
#gen-results-gallery .gallery,
#gen-results-gallery [class*="grid"] {
    overflow: visible !important;
    max-height: none !important;
}
#gen-results-gallery .overflow-auto,
#gen-results-gallery [style*="overflow"],
#agent-attempts-gallery .overflow-auto,
#agent-attempts-gallery [style*="overflow"] {
    overflow: visible !important;
}
#agent-attempts-gallery,
#agent-attempts-gallery > .block,
#agent-attempts-gallery .grid-wrap,
#agent-attempts-gallery .gallery,
#agent-attempts-gallery [class*="grid"] {
    overflow: visible !important;
    max-height: none !important;
}
#agent-results-row {
    align-items: stretch !important;
    gap: 12px !important;
}
/* 2 превью в 1 ряд (итерация 2) — та же высота, что при 3–4 (2 ряда) */
#agent-attempts-gallery .grid-container,
.gradio-container-6-10-0 #agent-attempts-gallery .grid-container {
    min-height: 216px !important;
    grid-auto-rows: minmax(100px, auto) !important;
}
#agent-attempts-col > .form,
#agent-attempts-col > .block {
    height: 100% !important;
}
/* height="auto" в Gradio всё равно вешает .fixed-height (min-height 320–450px) */
#gen-results-gallery .grid-wrap.fixed-height,
#gen-results-gallery .grid-wrap[class*="fixed-height"],
#agent-attempts-gallery .grid-wrap.fixed-height,
#agent-attempts-gallery .grid-wrap[class*="fixed-height"],
.gradio-container-6-10-0 #gen-results-gallery .grid-wrap.fixed-height,
.gradio-container-6-10-0 #agent-attempts-gallery .grid-wrap.fixed-height {
    min-height: 0 !important;
    max-height: none !important;
    height: auto !important;
}
#gen-results-gallery .gallery-container,
#agent-attempts-gallery .gallery-container,
.gradio-container-6-10-0 #gen-results-gallery .gallery-container,
.gradio-container-6-10-0 #agent-attempts-gallery .gallery-container {
    height: auto !important;
    min-height: 0 !important;
}
#gen-results-gallery .grid-container,
#agent-attempts-gallery .grid-container,
.gradio-container-6-10-0 #gen-results-gallery .grid-container,
.gradio-container-6-10-0 #agent-attempts-gallery .grid-container {
    grid-template-rows: repeat(var(--grid-rows), auto) !important;
    grid-auto-rows: auto !important;
    align-content: start !important;
}

/* Просмотр: панель настроек */
.preview-controls-panel {
    gap: 10px !important;
    padding: 12px !important;
    border: 1px solid var(--border-color-primary, #e2e8f0) !important;
    border-radius: 10px !important;
    background: var(--background-fill-secondary, #f8fafc) !important;
    margin-bottom: 10px !important;
}
.preview-actions-block > .block:first-child button {
    width: 100% !important;
    min-height: 2.75rem !important;
    border-radius: 8px !important;
}

/* Просмотр: без обрезания левой колонки */
#tab-preview { scroll-margin-top: 0 !important; }
.preview-tab-row {
    align-items: flex-start !important;
    overflow: visible !important;
}
.preview-tab-col {
    overflow: visible !important;
    min-height: 0 !important;
}
#preview-viewer-wrap {
    height: min(62vh, 520px) !important;
    min-height: min(62vh, 520px) !important;
    max-height: min(62vh, 520px) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    padding: 0 !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
}
#preview-viewer-wrap .html-container,
#preview-viewer-wrap .prose,
#preview-viewer-wrap > .wrap,
#preview-viewer-wrap > div {
    height: 100% !important;
    min-height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    border-radius: inherit !important;
}
#preview-viewer-wrap iframe {
    display: block !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 100% !important;
    max-height: none !important;
    border: none !important;
    border-radius: 0 !important;
}
#preview-viewer-wrap #pbr-viewer-placeholder,
#preview-viewer-wrap .preview-viewer-loading {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
    min-height: min(62vh, 520px) !important;
    max-height: none !important;
    border: none !important;
    border-radius: 0 !important;
}

/* 3D: без скачивания .glb */
.model3d-no-dl button[title="Download"],
.model3d-no-dl button[aria-label="Download"] {
    display: none !important;
}

/* ── General polish ──────────────────────────────── */
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
}
.gr-button-primary:hover { opacity: 0.9 !important; }
.tab-nav button {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
}
.tab-nav button.selected {
    border-bottom: 3px solid #667eea !important;
    font-weight: 700 !important;
}
"""

FORCE_LIGHT_JS = """
(function () {
  function forceLight() {
    document.documentElement.classList.remove("dark");
    document.querySelectorAll(".dark").forEach(function (el) {
      el.classList.remove("dark");
    });
  }
  forceLight();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", forceLight);
  }
  new MutationObserver(forceLight).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
    subtree: true,
  });
  var mq = window.matchMedia("(prefers-color-scheme: dark)");
  if (mq.addEventListener) {
    mq.addEventListener("change", forceLight);
  } else if (mq.addListener) {
    mq.addListener(forceLight);
  }
})();
"""

CUSTOM_HEAD = (
    """
<script>
"""
    + FORCE_LIGHT_JS
    + """
</script>
<style>
  .gradio-container { max-width: 100% !important; }
  .gradio-container-6-10-0 #ref-image-wrap .placeholder,
  .gradio-container-6-10-0 #ref-image-wrap [class*="placeholder"],
  .gradio-container-6-10-0 #ref-image-upload,
  .gradio-container-6-10-0 #ref-image-upload.block {
    border: none !important;
    border-width: 0 !important;
    border-style: none !important;
    outline: none !important;
    box-shadow: none !important;
  }
</style>
"""
    + """
<script>
(function () {
  function clearRefBorders() {
    var wrap = document.getElementById("ref-image-wrap");
    if (!wrap) return;
    wrap.querySelectorAll(
      ".placeholder, [class*='placeholder'], #ref-image-upload, button"
    ).forEach(function (el) {
      el.style.setProperty("border", "none", "important");
      el.style.setProperty("border-width", "0", "important");
      el.style.setProperty("border-style", "none", "important");
      el.style.setProperty("outline", "none", "important");
      el.style.setProperty("box-shadow", "none", "important");
    });
  }
  function attach() {
    var wrap = document.getElementById("ref-image-wrap");
    if (!wrap || wrap.dataset.borderFix) return;
    wrap.dataset.borderFix = "1";
    clearRefBorders();
    new MutationObserver(clearRefBorders).observe(wrap, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class", "style"],
    });
  }
  window.addEventListener("load", function () {
    attach();
    var n = 0;
    var iv = setInterval(function () {
      attach();
      if (++n > 40) clearInterval(iv);
    }, 250);
  });
})();
</script>
<script>
"""
    + PBR_VIEWER_HEAD_JS
    + """
</script>
"""
)

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=["system-ui", "Segoe UI", "sans-serif"],
    font_mono=["Consolas", "monospace"],
).set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#f8fafc",
    body_text_color="*neutral_800",
    body_text_color_dark="*neutral_800",
    background_fill_primary="white",
    background_fill_primary_dark="white",
    background_fill_secondary="#f8fafc",
    background_fill_secondary_dark="#f8fafc",
    block_background_fill="white",
    block_background_fill_dark="white",
    block_border_width="0px",
    block_shadow="0 1px 3px rgba(0,0,0,0.08)",
    block_label_background_fill="white",
    block_label_background_fill_dark="white",
    block_label_text_color="*neutral_700",
    block_label_text_color_dark="*neutral_700",
    block_label_border_color="*neutral_200",
    block_label_border_color_dark="*neutral_200",
    input_background_fill="white",
    input_background_fill_dark="white",
    border_color_primary="*neutral_200",
    border_color_primary_dark="*neutral_200",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)

with gr.Blocks(title="TextureGen AI", theme=theme) as demo:

    with gr.Row(elem_classes="title-row"):
        gr.Markdown("# TextureGen AI")

    with gr.Tab("Генерация"):
        with gr.Row():
            with gr.Column(scale=2, min_width=500, elem_classes="fixed-left"):
                gen_mode = gr.Radio(
                    ["По референсу", "По описанию"],
                    value="По референсу",
                    label="Режим",
                )
                with gr.Group(visible=True, elem_id="ref-image-wrap") as ref_group:
                    ref_image = gr.Image(
                        type="pil", label="Референсное изображение",
                        format="png", sources=["upload", "clipboard"],
                        height=260, buttons=["download"],
                        elem_id="ref-image-upload",
                        elem_classes="no-upload-border",
                    )
                with gr.Group(visible=False) as txt_group:
                    txt_prompt = gr.Textbox(
                        label="Описание текстуры (на английском)",
                        placeholder="old brick wall with cracks, weathered surface",
                        lines=3,
                    )
                with gr.Column(elem_id="ref-gen-controls") as ref_controls:
                    ref_material = gr.Dropdown(
                        choices=MATERIAL_CHOICES,
                        value="Авто",
                        label="Тип материала",
                        filterable=False,
                    )
                    with gr.Row(elem_id="strength-wrap"):
                        strength = gr.Slider(
                            0.3, 1.0, value=0.65, step=0.05,
                            label="Сила изменения",
                        )
                    with gr.Row():
                        ref_guidance = gr.Slider(
                            1, 20, value=7.5, step=0.5,
                            label="Точность следования",
                        )
                        ref_steps = gr.Slider(
                            6, 32, value=19, step=1,
                            label="Кол-во шагов",
                        )
                    ref_num_imgs = gr.Slider(
                        1, 4, value=2, step=1, label="Кол-во генераций",
                    )
                with gr.Column(
                    elem_id="txt-gen-controls",
                    elem_classes=["gen-panel-hidden"],
                ) as txt_controls:
                    with gr.Row():
                        txt_guidance = gr.Slider(
                            1, 20, value=7.5, step=0.5,
                            label="Точность следования",
                        )
                        txt_steps = gr.Slider(
                            6, 32, value=19, step=1,
                            label="Кол-во шагов",
                        )
                    txt_num_imgs = gr.Slider(
                        1, 4, value=2, step=1, label="Кол-во генераций",
                    )
                gen_btn = gr.Button("Сгенерировать", variant="primary", size="lg")
            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    label="Результаты генерации",
                    columns=GEN_GALLERY_COLUMNS,
                    height="auto",
                    format="png",
                    interactive=False,
                    buttons=["download"],
                    elem_id="gen-results-gallery",
                )

        gen_mode.change(
            _toggle_gen_mode,
            inputs=[gen_mode],
            outputs=[ref_group, txt_group, ref_controls, txt_controls],
        )
        gen_event = gen_btn.click(
            fn_generate_unified,
            inputs=[
                gen_mode, ref_image, txt_prompt,
                ref_material,
                strength,
                ref_guidance, ref_steps, ref_num_imgs,
                txt_guidance, txt_steps, txt_num_imgs,
            ],
            outputs=gallery,
        )

    with gr.Tab("ML-Агент"):
        with gr.Row():
            with gr.Column(scale=2, min_width=500, elem_classes="fixed-left"):
                agent_ref = gr.Image(
                    type="pil", label="Референсное изображение",
                    format="png", sources=["upload", "clipboard"],
                    height=280, buttons=["download"],
                )
                agent_material = gr.Dropdown(
                    choices=MATERIAL_CHOICES,
                    value="Авто",
                    label="Тип материала",
                    filterable=False,
                )
                with gr.Row():
                    agent_iters = gr.Slider(
                        2, 5, value=3, step=1,
                        label="Макс. итераций",
                    )
                    agent_thresh = gr.Slider(
                        0.5, 0.95, value=0.70, step=0.05,
                        label="Порог качества",
                    )
                agent_btn = gr.Button("Запустить агент", variant="primary", size="lg")
                agent_report = gr.Textbox(
                    label="Отчёт агента", lines=10, interactive=False,
                )
            with gr.Column(scale=3):
                with gr.Row(elem_id="agent-results-row"):
                    with gr.Column(scale=6, min_width=320, elem_id="agent-best-col"):
                        agent_best = gr.Image(
                            type="pil", label="Лучший результат",
                            format="png", interactive=False, buttons=["download"],
                        )
                    with gr.Column(scale=4, min_width=220, elem_id="agent-attempts-col"):
                        agent_gallery = gr.Gallery(
                            label="Все попытки агента",
                            columns=2,
                            height="auto",
                            format="png",
                            interactive=False,
                            buttons=["download"],
                            elem_id="agent-attempts-gallery",
                        )

        agent_btn.click(
            fn_agent_run,
            inputs=[agent_ref, agent_material, agent_iters, agent_thresh],
            outputs=[agent_best, agent_gallery, agent_report],
        )

    with gr.Tab("Просмотр", elem_id="tab-preview"):
        preview_pbr_maps = gr.State(None)
        preview_seamless_active = gr.State(False)
        preview_glb_url = gr.Textbox("", visible=False, elem_id="preview-glb-url")
        with gr.Row(elem_classes="preview-tab-row"):
            with gr.Column(
                scale=2, min_width=500,
                elem_classes=["fixed-left", "preview-tab-col"],
            ):
                preview_diffuse = gr.Image(
                    type="pil", label="Текстура",
                    format="png", sources=["upload", "clipboard"],
                    height=240, buttons=["download"],
                )
                with gr.Group(elem_classes="preview-controls-panel"):
                    preview_mode = gr.Radio(
                        PBR_MODE_RU,
                        value=PBR_MODE_RU[0],
                        label="PBR-карта",
                    )
                    preview_shape = gr.Radio(
                        list(SHAPE_ORDER_RU),
                        value=DEFAULT_SHAPE_RU,
                        label="Геометрия",
                    )
                    preview_seamless = gr.Radio(
                        [SEAMLESS_OFF, SEAMLESS_ON],
                        value=SEAMLESS_OFF,
                        label="Бесшовная обработка",
                    )
                    preview_tile = gr.Slider(
                        1, 6, value=1, step=1,
                        label="Тайлинг",
                    )
                    preview_hdr = gr.Radio(
                        HDR_LABELS,
                        value=HDR_LABELS[0],
                        label="HDR сцена",
                    )
                with gr.Column(elem_classes="preview-actions-block"):
                    preview_btn = gr.Button(
                        "Показать на модели",
                        variant="primary",
                        size="lg",
                    )
                    with gr.Row(
                        elem_id="preview-actions-wrap",
                        elem_classes=["preview-actions", "preview-actions--hidden"],
                    ) as preview_actions:
                        preview_dl_one = gr.DownloadButton(
                            "Скачать карту",
                            value=fn_dl_current_pbr,
                            inputs=[preview_pbr_maps, preview_mode],
                            variant="primary",
                            size="lg",
                            scale=1,
                        )
                        preview_dl_all = gr.DownloadButton(
                            "Скачать все PBR",
                            value=fn_dl_all_pbr,
                            inputs=[preview_pbr_maps],
                            variant="primary",
                            size="lg",
                            scale=1,
                        )
                        preview_dl_tile = gr.DownloadButton(
                            "Скачать тайлинг",
                            value=fn_dl_tiling,
                            inputs=[
                                preview_diffuse,
                                preview_tile,
                                preview_pbr_maps,
                                preview_seamless_active,
                            ],
                            variant="primary",
                            size="lg",
                            scale=1,
                            elem_id="preview-dl-tile",
                            elem_classes=["tile-dl-hidden"],
                        )
            with gr.Column(scale=3, elem_classes="preview-viewer-col"):
                gr.Markdown("**3D-превью**")
                preview_viewer = gr.HTML(
                    viewer_placeholder_html(),
                    elem_id="preview-viewer-wrap",
                )

        _preview_inputs = [
            preview_diffuse,
            preview_shape,
            preview_tile,
            preview_mode,
            preview_pbr_maps,
        ]
        _preview_update_inputs = _preview_inputs + [preview_hdr, preview_seamless_active]

        preview_btn.click(
            fn_preview3d,
            inputs=_preview_inputs[:4] + [preview_hdr, preview_seamless_active],
            outputs=[
                preview_pbr_maps,
                preview_actions,
                preview_dl_tile,
                preview_viewer,
            ],
            show_progress="hidden",
        ).then(js=PREVIEW_SHOW_DOWNLOADS_JS)
        preview_seamless.change(
            fn_preview_seamless_changed,
            inputs=[
                preview_seamless,
                preview_diffuse,
                preview_shape,
                preview_tile,
                preview_mode,
                preview_pbr_maps,
                preview_hdr,
            ],
            outputs=[
                preview_seamless_active,
                preview_seamless,
                preview_pbr_maps,
                preview_actions,
                preview_dl_tile,
                preview_viewer,
            ],
            show_progress="hidden",
        )
        preview_diffuse.change(
            fn_preview_diffuse_changed,
            inputs=[preview_diffuse],
            outputs=[preview_seamless_active, preview_seamless],
            show_progress="hidden",
        ).then(js=PREVIEW_HIDE_DOWNLOADS_JS)
        _preview_live_outputs = [preview_glb_url, preview_viewer]
        for ctrl in (preview_mode, preview_shape):
            ctrl.change(
                fn_preview_update,
                inputs=_preview_update_inputs,
                outputs=_preview_live_outputs,
                show_progress="hidden",
            )
        preview_tile.change(
            fn_preview_update,
            inputs=_preview_update_inputs,
            outputs=_preview_live_outputs,
            show_progress="hidden",
        )
        for tile_evt in (preview_tile.change, preview_tile.input):
            tile_evt(
                fn_tile_dl_button_update,
                inputs=[preview_tile],
                outputs=[preview_dl_tile],
                show_progress="hidden",
            )
        preview_hdr.change(
            None,
            inputs=[preview_hdr],
            js=HDR_SWITCH_JS,
            show_progress="hidden",
        )


def _find_free_port(start: int = 7860, attempts: int = 20) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start


if __name__ == "__main__":
    port = _find_free_port()
    if port != 7860:
        logger.warning("Порт 7860 занят, используется %s", port)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=CUSTOM_CSS,
        head=CUSTOM_HEAD,
        allowed_paths=[
            str(_ASSETS_DIR),
            str(_PREVIEW_EXPORT_DIR),
            str(Path(__file__).resolve().parent / "models"),
        ],
    )
