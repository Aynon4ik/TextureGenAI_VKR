import logging
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from skimage.metrics import structural_similarity as ssim

from src.classifier import TextureClassifier
from src.generator import TextureGenerator
from src.materials import CLASSIFIER_AUTO_MIN_CONFIDENCE

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    best_image: Image.Image
    best_score: float
    best_params: dict
    all_attempts: list[dict] = field(default_factory=list)
    material_type: str = "default"
    material_confidence: float = 0.0
    material_confident: bool = True
    iterations_used: int = 0


def _image_to_gray_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float64) / 255.0


def compute_ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    size = (256, 256)
    a = _image_to_gray_array(img_a.resize(size))
    b = _image_to_gray_array(img_b.resize(size))
    return float(ssim(a, b, data_range=1.0))


def compute_color_histogram_score(reference: Image.Image, generated: Image.Image) -> float:
    size = (256, 256)
    ref = np.array(reference.resize(size).convert("RGB"), dtype=np.float64)
    gen = np.array(generated.resize(size).convert("RGB"), dtype=np.float64)

    score = 0.0
    for ch in range(3):
        h_ref, _ = np.histogram(ref[:, :, ch], bins=64, range=(0, 256))
        h_gen, _ = np.histogram(gen[:, :, ch], bins=64, range=(0, 256))
        h_ref = h_ref / (h_ref.sum() + 1e-8)
        h_gen = h_gen / (h_gen.sum() + 1e-8)
        score += np.sum(np.minimum(h_ref, h_gen))
    return score / 3.0


def compute_texture_complexity(image: Image.Image) -> float:
    from scipy.ndimage import laplace
    gray = _image_to_gray_array(image.resize((256, 256)))
    lap = laplace(gray)
    return float(np.var(lap))


def compute_seamless_score(image: Image.Image, border: int = 16) -> float:
    arr = np.array(image.convert("RGB"), dtype=np.float64) / 255.0
    h, w, _ = arr.shape

    top, bottom = arr[:border, :, :], arr[-border:, :, :]
    left, right = arr[:, :border, :], arr[:, -border:, :]

    v_diff = np.mean(np.abs(top - bottom[::-1, :, :]))
    h_diff = np.mean(np.abs(left - right[:, ::-1, :]))

    return float(1.0 - (v_diff + h_diff) / 2.0)


def compute_quality_score(
    reference: Image.Image,
    generated: Image.Image,
    w_color: float = 0.35,
    w_complexity: float = 0.25,
    w_seamless: float = 0.25,
    w_ssim: float = 0.15,
) -> tuple[float, dict]:
    s_color = compute_color_histogram_score(reference, generated)
    s_complexity = min(compute_texture_complexity(generated) / 0.005, 1.0)
    s_seamless = compute_seamless_score(generated)
    s_ssim = compute_ssim(reference, generated)

    total = (
        w_color * s_color
        + w_complexity * s_complexity
        + w_seamless * s_seamless
        + w_ssim * s_ssim
    )

    breakdown = {
        "color_match": round(s_color, 4),
        "complexity": round(s_complexity, 4),
        "seamless": round(s_seamless, 4),
        "ssim": round(s_ssim, 4),
        "total": round(total, 4),
    }
    return total, breakdown


_PARAM_STRATEGIES = [
    {"strength": 0.55, "guidance_scale": 7.0, "label": "conservative"},
    {"strength": 0.65, "guidance_scale": 7.5, "label": "balanced"},
    {"strength": 0.75, "guidance_scale": 8.0, "label": "creative"},
    {"strength": 0.50, "guidance_scale": 9.0, "label": "high-guidance"},
    {"strength": 0.80, "guidance_scale": 6.0, "label": "low-guidance"},
]


class TextureAgent:
    def __init__(
        self,
        generator: TextureGenerator,
        classifier: TextureClassifier | None = None,
        quality_threshold: float = 0.70,
        max_iterations: int = 5,
    ):
        self.generator = generator
        self.classifier = classifier
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations

    def _classify_material(self, image: Image.Image) -> tuple[str, float, bool]:
        if self.classifier is None:
            return "default", 0.0, False
        material, conf, confident, _ = self.classifier.predict_for_auto(
            image, min_confidence=CLASSIFIER_AUTO_MIN_CONFIDENCE
        )
        return material, conf, confident

    def _resolve_material(
        self,
        reference_image: Image.Image,
        material_override: str | None,
    ) -> tuple[str, float, bool]:
        if material_override and material_override != "default":
            logger.info("Agent: material overridden to %s", material_override)
            return material_override, 1.0, True
        material, confidence, material_confident = self._classify_material(reference_image)
        logger.info(
            "Agent: detected material=%s (%.1f%%, confident=%s)",
            material, confidence * 100, material_confident,
        )
        return material, confidence, material_confident

    def iter_run(
        self,
        reference_image: Image.Image,
        custom_prompt: str = "",
        seed_base: int = -1,
        material_override: str | None = None,
        *,
        inference_steps: int = 24,
    ):
        material, confidence, material_confident = self._resolve_material(
            reference_image, material_override
        )

        best_image = None
        best_score = -1.0
        best_params: dict = {}
        all_attempts: list[dict] = []

        for i, strategy in enumerate(_PARAM_STRATEGIES[: self.max_iterations]):
            seed = (seed_base + i * 1000) if seed_base >= 0 else -1

            logger.info(
                "Agent: iteration %d/%d strategy=%s strength=%.2f guidance=%.1f",
                i + 1, self.max_iterations, strategy["label"],
                strategy["strength"], strategy["guidance_scale"],
            )

            images = self.generator.generate_from_reference(
                reference_image=reference_image,
                material_type=material,
                custom_prompt=custom_prompt,
                strength=strategy["strength"],
                guidance_scale=strategy["guidance_scale"],
                num_inference_steps=inference_steps,
                num_images=1,
                seed=seed,
            )

            generated = images[0]

            score, breakdown = compute_quality_score(reference_image, generated)

            attempt = {
                "iteration": i + 1,
                "strategy": strategy["label"],
                "strength": strategy["strength"],
                "guidance_scale": strategy["guidance_scale"],
                "score": score,
                "metrics": breakdown,
                "image": generated,
            }
            all_attempts.append(attempt)

            logger.info("Agent: score=%.4f breakdown=%s", score, breakdown)

            if score > best_score:
                best_score = score
                best_image = generated
                best_params = {
                    "strategy": strategy["label"],
                    "strength": strategy["strength"],
                    "guidance_scale": strategy["guidance_scale"],
                    "seed": seed,
                }

            yield AgentResult(
                best_image=best_image,
                best_score=best_score,
                best_params=best_params,
                all_attempts=all_attempts,
                material_type=material,
                material_confidence=confidence,
                material_confident=material_confident,
                iterations_used=len(all_attempts),
            )

            if score >= self.quality_threshold:
                logger.info("Agent: quality threshold reached, stopping early")
                break

    def run(
        self,
        reference_image: Image.Image,
        custom_prompt: str = "",
        seed_base: int = -1,
        material_override: str | None = None,
    ) -> AgentResult:
        last: AgentResult | None = None
        for last in self.iter_run(
            reference_image,
            custom_prompt=custom_prompt,
            seed_base=seed_base,
            material_override=material_override,
        ):
            pass
        if last is None:
            raise RuntimeError("Agent produced no iterations")
        return last
