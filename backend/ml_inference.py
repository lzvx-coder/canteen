import base64
import csv
import io
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "food_classifier.pt"
CHINESEFOODNET_LABELS_PATH = BASE_DIR / "data" / "chinesefoodnet_labels.csv"

FOOD101_TO_DISH = {
    "dumplings": "饺子",
    "edamame": "毛豆",
    "eggs_benedict": "荷包蛋",
    "fried_rice": "扬州炒饭",
    "hot_and_sour_soup": "酸辣汤",
    "miso_soup": "酱汤",
    "omelette": "蛋包饭",
    "pizza": "披萨",
    "ramen": "红烧牛肉面",
    "seaweed_salad": "凉拌海带丝",
    "spring_rolls": "韭菜盒子",
    "sushi": "紫菜包饭",
}


def _decode_data_url(image_data: str) -> Any:
    from PIL import Image

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    raw = base64.b64decode(image_data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@lru_cache(maxsize=1)
def _load_label_map() -> dict[str, str]:
    label_map: dict[str, str] = {}
    if not CHINESEFOODNET_LABELS_PATH.exists():
        return label_map

    with CHINESEFOODNET_LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            class_id = str(row.get("id", "")).strip()
            name = str(row.get("name", "")).strip()
            if class_id and name:
                label_map[class_id] = name
                if class_id.isdigit():
                    label_map[str(int(class_id))] = name
    return label_map


def _map_class_to_dish(class_name: str) -> str:
    normalized = class_name.replace("\\", "/").split("/", 1)[0].strip()
    label_map = _load_label_map()
    if normalized in label_map:
        return label_map[normalized]
    if normalized.isdigit():
        padded = f"{int(normalized):03d}"
        if padded in label_map:
            return label_map[padded]
    return FOOD101_TO_DISH.get(class_name, class_name)


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, dict[str, Any]] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        import torch
        from torchvision import models
    except Exception:
        return None

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_food_image(image_data: str, top_k: int = 5) -> list[dict[str, object]]:
    loaded = _load_model()
    if not loaded or not image_data:
        return []

    import torch
    from torchvision import transforms

    model, checkpoint = loaded
    image_size = int(checkpoint.get("image_size", 224))
    class_names = checkpoint["class_names"]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = _decode_data_url(image_data)
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
    predictions: list[dict[str, object]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        class_name = class_names[index]
        predictions.append(
            {
                "label": class_name,
                "dish_name": _map_class_to_dish(class_name),
                "confidence": round(float(value), 4),
            }
        )
    return predictions
