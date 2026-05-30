import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.classifier_model import DEFAULT_ARCHITECTURE, build_classifier
from src.materials import CLASSIFIER_AUTO_MIN_CONFIDENCE, CLASSIFIER_CLASSES as CLASSES

logger = logging.getLogger(__name__)

__all__ = ["TextureClassifier", "CLASSES"]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _architecture_from_meta(model_path: str | None) -> str:
    if not model_path:
        return DEFAULT_ARCHITECTURE
    p = Path(model_path).with_name(Path(model_path).stem + "_meta.json")
    if p.is_file():
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            if "architecture" in meta:
                return meta["architecture"]
        except Exception:
            pass
    if model_path and Path(model_path).is_file():
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            if "fc.weight" in state:
                return "resnet18"
            if "classifier.weight" in state:
                return "densenet121"
        except Exception:
            pass
    return DEFAULT_ARCHITECTURE


class TextureClassifier:
    def __init__(self, model_path: str | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.architecture = _architecture_from_meta(model_path)
        self.model, _ = build_classifier(len(CLASSES), architecture=self.architecture, pretrained=False)
        self.fine_tuned = False

        if model_path:
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self.fine_tuned = True
                logger.info("Loaded classifier (%s) from %s", self.architecture, model_path)
            except Exception as exc:
                logger.warning(
                    "Could not load weights from %s: %s — using random init",
                    model_path,
                    exc,
                )

        self.model = self.model.to(self.device)
        self.model.eval()

    def classify(self, image: Image.Image) -> dict[str, float]:
        tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        return {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}

    def predict(self, image: Image.Image) -> str:
        scores = self.classify(image)
        return max(scores, key=scores.get)

    def predict_for_auto(
        self,
        image: Image.Image,
        min_confidence: float = CLASSIFIER_AUTO_MIN_CONFIDENCE,
    ) -> tuple[str, float, bool, dict[str, float]]:
        scores = self.classify(image)
        best = max(scores, key=scores.get)
        conf = float(scores[best])
        if conf < min_confidence:
            return "default", conf, False, scores
        return best, conf, True, scores
