from __future__ import annotations

import logging
import math
from collections.abc import Callable

import torch
from PIL import Image
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)

logger = logging.getLogger(__name__)

MATERIAL_PROMPTS = {
    "tiles": "ceramic tiles texture, seamless, PBR material, high resolution, detailed surface, top-down view",
    "metal": "metal surface texture, seamless, PBR material, high resolution, metallic sheen, top-down view",
    "wood": "wood grain texture, seamless, PBR material, high resolution, natural wood pattern, top-down view",
    "fabrics": "fabric cloth texture, seamless, PBR material, high resolution, woven textile, top-down view",
    "paper": "paper surface texture, seamless, PBR material, high resolution, paper grain, top-down view",
    "concrete": "concrete wall texture, seamless, PBR material, high resolution, raw concrete, top-down view",
    "ground": "ground terrain texture, seamless, PBR material, high resolution, natural soil, top-down view",
    "marble": "marble stone texture, seamless, PBR material, high resolution, natural marble veins, top-down view",
    "bricks": (
        "red clay brick wall texture, running bond masonry, staggered courses, "
        "horizontal brick rows, offset vertical joints, straight mortar lines, "
        "rectangular bricks, flat wall surface, orthographic front view, even diffuse lighting, "
        "seamless tileable, PBR material, high resolution, sharp joints, no cushion, no upholstery"
    ),
    "default": "surface material texture, seamless, PBR material, high resolution, detailed, top-down view",
}

_NEGATIVE_CORE = (
    "face, head, portrait, person, human, man, woman, body, bust, animal, "
    "blurry, ugly, low quality, jpeg artifacts, watermark, text, logo, "
    "deformed, cropped, out of frame, "
    "sphere, cylinder, ball, 3d object, 3d render, "
    "collage, split screen, multiple panels, product photo"
)
_NEGATIVE_GRID = ", grid"

NEGATIVE_PROMPT = _NEGATIVE_CORE + _NEGATIVE_GRID

_MATERIAL_NEGATIVE_NO_GRID = frozenset({"bricks", "tiles"})


def negative_prompt_for_material(material_type: str) -> str:
    if material_type in _MATERIAL_NEGATIVE_NO_GRID:
        return _NEGATIVE_CORE
    return NEGATIVE_PROMPT


ProgressFn = Callable[[float, str], None] | None


def img2img_denoise_steps(num_inference_steps: int, strength: float) -> int:
    steps = int(num_inference_steps)
    return max(1, min(int(steps * float(strength)), steps))


def denoise_steps_to_inference(denoise_steps: int, strength: float) -> int:
    s = max(1e-6, float(strength))
    return max(1, int(math.ceil(int(denoise_steps) / s)))


def make_denoise_progress_callback(
    on_progress: ProgressFn,
    total_steps: int,
    *,
    phase_start: float = 0.08,
    phase_end: float = 0.98,
    simple_desc: bool = False,
):
    completed = [0]

    def callback_on_step_end(pipe, step_index, _timestep, callback_kwargs):
        order = int(getattr(pipe.scheduler, "order", 1) or 1)
        timesteps_total = int(getattr(pipe, "_num_timesteps", total_steps * order))
        num_warmup = timesteps_total - total_steps * order
        is_last = step_index >= timesteps_total - 1
        tick = is_last or (
            (step_index + 1) > num_warmup and (step_index + 1) % order == 0
        )
        if not tick:
            return callback_kwargs

        completed[0] = min(completed[0] + 1, total_steps)
        cur = completed[0]
        pct = int(round(100 * cur / max(total_steps, 1)))
        if simple_desc:
            frac = cur / max(total_steps, 1)
            desc = "Генерация текстур…"
        else:
            frac = phase_start + (cur / max(total_steps, 1)) * (phase_end - phase_start)
            desc = f"Генерация текстур… — {pct}% ({cur}/{total_steps})"
        if on_progress:
            on_progress(min(frac, 1.0), desc)
        return callback_kwargs

    return callback_on_step_end


class TextureGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_id = model_id
        self.img2img_pipe = None
        self.txt2img_pipe = None

    def load_model(self):
        logger.info("Loading Stable Diffusion img2img pipeline (%s)...", self.model_id)
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.img2img_pipe.scheduler.config
        )
        self.img2img_pipe = self.img2img_pipe.to(self.device)

        if self.device == "cuda":
            self.img2img_pipe.enable_attention_slicing()

        logger.info("img2img pipeline ready on %s", self.device)

    def load_txt2img(self):
        logger.info("Loading Stable Diffusion txt2img pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.txt2img_pipe.scheduler.config
        )
        self.txt2img_pipe = self.txt2img_pipe.to(self.device)

        if self.device == "cuda":
            self.txt2img_pipe.enable_attention_slicing()

        logger.info("txt2img pipeline ready on %s", self.device)

    def _build_prompt(self, material_type: str, custom_prompt: str) -> str:
        base = MATERIAL_PROMPTS.get(material_type, MATERIAL_PROMPTS["default"])
        if custom_prompt and custom_prompt.strip():
            return f"{custom_prompt.strip()}, {base}"
        return base

    def generate_from_reference(
        self,
        reference_image: Image.Image,
        material_type: str = "default",
        custom_prompt: str = "",
        strength: float = 0.65,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int = -1,
        size: int = 512,
        on_progress: ProgressFn = None,
    ) -> list[Image.Image]:
        if self.img2img_pipe is None:
            self.load_model()

        ref = reference_image.convert("RGB").resize((size, size), Image.LANCZOS)
        prompt = self._build_prompt(material_type, custom_prompt)

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        neg = negative_prompt_for_material(material_type)
        denoise_steps = img2img_denoise_steps(num_inference_steps, strength)
        step_cb = (
            make_denoise_progress_callback(on_progress, denoise_steps, simple_desc=True)
            if on_progress
            else None
        )
        self.img2img_pipe.set_progress_bar_config(disable=bool(on_progress))

        results = self.img2img_pipe(
            prompt=prompt,
            image=ref,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images,
            negative_prompt=neg,
            generator=generator,
            callback_on_step_end=step_cb,
        ).images

        return results

    def generate_from_text(
        self,
        prompt: str,
        material_type: str = "default",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int = -1,
        size: int = 512,
        on_progress: ProgressFn = None,
    ) -> list[Image.Image]:
        if self.txt2img_pipe is None:
            self.load_txt2img()

        full_prompt = self._build_prompt(material_type, prompt)

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        neg = negative_prompt_for_material(material_type)
        denoise_steps = max(1, int(num_inference_steps))
        step_cb = (
            make_denoise_progress_callback(on_progress, denoise_steps, simple_desc=True)
            if on_progress
            else None
        )
        self.txt2img_pipe.set_progress_bar_config(disable=bool(on_progress))

        results = self.txt2img_pipe(
            prompt=full_prompt,
            negative_prompt=neg,
            width=size,
            height=size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images,
            generator=generator,
            callback_on_step_end=step_cb,
        ).images

        return results
