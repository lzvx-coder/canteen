import argparse
import random
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_DATA = BASE_DIR / "data" / "food_images"
DEFAULT_FOOD101_ROOT = BASE_DIR / "data" / "food101"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "food_classifier.pt"


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def limit_imagefolder_samples(source_dir: Path, target_dir: Path, max_per_class: int) -> Path:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        images = [path for path in class_dir.rglob("*") if path.suffix.lower() in image_exts]
        if not images:
            continue
        random.shuffle(images)
        dst_class = target_dir / class_dir.name
        dst_class.mkdir(parents=True, exist_ok=True)
        for image_path in images[:max_per_class]:
            shutil.copy2(image_path, dst_class / image_path.name)
    return target_dir


def make_dataset(args: argparse.Namespace, train_transform: transforms.Compose, eval_transform: transforms.Compose):
    if args.source == "food101":
        train_dataset = datasets.Food101(
            root=str(args.food101_root),
            split="train",
            transform=train_transform,
            download=True,
        )
        val_dataset = datasets.Food101(
            root=str(args.food101_root),
            split="test",
            transform=eval_transform,
            download=True,
        )
        class_names = list(train_dataset.classes)
        if args.max_classes:
            allowed = set(class_names[: args.max_classes])
            train_dataset._image_files = [p for p in train_dataset._image_files if p.parts[-2] in allowed]
            train_dataset._labels = [class_names.index(p.parts[-2]) for p in train_dataset._image_files]
            val_dataset._image_files = [p for p in val_dataset._image_files if p.parts[-2] in allowed]
            val_dataset._labels = [class_names.index(p.parts[-2]) for p in val_dataset._image_files]
            class_names = class_names[: args.max_classes]
        return train_dataset, val_dataset, class_names

    data_dir = Path(args.data_dir)
    if args.max_per_class:
        data_dir = limit_imagefolder_samples(data_dir, BASE_DIR / "data" / "_limited_food_images", args.max_per_class)
    full_train = datasets.ImageFolder(str(data_dir), transform=train_transform)
    full_eval = datasets.ImageFolder(str(data_dir), transform=eval_transform)
    val_size = max(1, int(len(full_train) * args.val_ratio))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)
    _, val_eval_subset = random_split(full_eval, [train_size, val_size], generator=generator)
    return train_subset, val_eval_subset, list(full_train.classes)


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, training: bool) -> tuple[float, float]:
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch food image classifier.")
    parser.add_argument("--source", choices=["local", "food101"], default="local")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_LOCAL_DATA)
    parser.add_argument("--food101-root", type=Path, default=DEFAULT_FOOD101_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset, val_dataset, class_names = make_dataset(args, train_transform, eval_transform)
    if len(class_names) < 2:
        raise ValueError("至少需要两个类别文件夹才能训练分类模型。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, classes={len(class_names)}, train={len(train_dataset)}, val={len(val_dataset)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "source": args.source,
                    "val_acc": best_acc,
                },
                args.output,
            )
            print(f"saved best model -> {args.output}")


if __name__ == "__main__":
    main()
