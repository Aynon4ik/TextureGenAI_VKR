from __future__ import annotations

CLASSIFIER_CLASSES: tuple[str, ...] = (
    "tiles",
    "metal",
    "wood",
    "fabrics",
    "bricks",
)

CLASSIFIER_AUTO_MIN_CONFIDENCE: float = 0.55

MATERIAL_LABELS_RU: dict[str, str] = {
    "default": "Авто",
    "tiles": "Плитка",
    "metal": "Металл",
    "wood": "Дерево",
    "fabrics": "Ткань",
    "paper": "Бумага",
    "bricks": "Кирпич",
    "concrete": "Бетон",
    "ground": "Грунт",
    "marble": "Мрамор",
}

MATERIAL_UI_ORDER: tuple[str, ...] = (
    "default",
    "bricks",
    "wood",
    "metal",
    "tiles",
    "fabrics",
    "paper",
    "concrete",
    "ground",
    "marble",
)


def material_dropdown_choices() -> list[str]:
    return [MATERIAL_LABELS_RU[k] for k in MATERIAL_UI_ORDER]
