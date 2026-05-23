from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("backend/data/food_images")
MAX_CLASSES = 30
MAX_PER_CLASS = 120

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading dataset...")
dataset = load_dataset("chaeso/food_chinese_2017", split="train")

print(dataset)
print(dataset.features)

label_feature = dataset.features["label"]
class_names = label_feature.names

selected_class_ids = list(range(min(MAX_CLASSES, len(class_names))))
counts = {class_id: 0 for class_id in selected_class_ids}

print(f"Selected classes: {len(selected_class_ids)}")
for class_id in selected_class_ids:
    print(class_id, class_names[class_id])

for index, item in enumerate(dataset):
    label_id = int(item["label"])

    if label_id not in counts:
        continue

    if counts[label_id] >= MAX_PER_CLASS:
        continue

    class_name = class_names[label_id]
    class_dir = OUTPUT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    image = item["image"].convert("RGB")
    image_path = class_dir / f"{counts[label_id]:04d}.jpg"
    image.save(image_path, quality=92)

    counts[label_id] += 1

    if index % 500 == 0:
        total = sum(counts.values())
        print(f"Processed {index}, saved {total} images...")

    if all(count >= MAX_PER_CLASS for count in counts.values()):
        break

print("Done.")
print("Saved to:", OUTPUT_DIR.resolve())
for class_id in selected_class_ids:
    print(class_names[class_id], counts[class_id])