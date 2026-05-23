from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRICE_RE = re.compile(r"(?P<price>\d+(?:\.\d+)?)\s*(?:元|块|/|／|每|个|份|两)?")
SKIP_WORDS = {
    "元",
    "个",
    "份",
    "每",
    "可添加",
    "米饭",
    "一元",
    "二楼",
    "自选区",
    "风味",
    "食堂",
}


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTS:
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def make_ocr() -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit(
            "缺少 PaddleOCR。请先安装：\n"
            "python -m pip install paddleocr paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc

    try:
        return PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    except TypeError:
        try:
            return PaddleOCR(use_textline_orientation=True, lang="ch")
        except TypeError:
            return PaddleOCR(lang="ch")


def flatten_paddle_result(result: Any) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            texts = node.get("rec_texts")
            scores = node.get("rec_scores")
            if isinstance(texts, list):
                for index, text in enumerate(texts):
                    score = 0.0
                    if isinstance(scores, list) and index < len(scores):
                        try:
                            score = float(scores[index])
                        except (TypeError, ValueError):
                            score = 0.0
                    rows.append((str(text), score))
                return
            for value in node.values():
                visit(value)
            return

        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[1], (list, tuple)) and len(node[1]) >= 2:
                text, score = node[1][0], node[1][1]
                if isinstance(text, str):
                    try:
                        rows.append((text, float(score)))
                    except (TypeError, ValueError):
                        rows.append((text, 0.0))
                    return
            for value in node:
                visit(value)

    visit(result)
    return rows


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("￥", "")
    text = text.replace("¥", "")
    text = re.sub(r"\s+", "", text)
    return text


def looks_like_dish(text: str) -> bool:
    if len(text) < 2:
        return False
    if text in SKIP_WORDS:
        return False
    if text.isdigit():
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?元?/?[个份两]?", text):
        return False
    if any(word in text for word in ["电话", "微信", "扫码", "优惠", "欢迎", "价格", "菜单"]):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def split_menu_line(text: str) -> list[tuple[str, str]]:
    text = clean_text(text)
    if not text:
        return []

    matches = list(PRICE_RE.finditer(text))
    if not matches:
        return [(text, "")]

    rows: list[tuple[str, str]] = []
    previous_end = 0
    previous_price = ""
    for match in matches:
        price = match.group("price")
        name = text[previous_end : match.start()]
        name = re.sub(r"[：:·.\-—]+$", "", name)
        name = name.strip()
        if looks_like_dish(name):
            rows.append((name, price))
        previous_end = match.end()
        previous_price = price

    tail = text[previous_end:].strip()
    if looks_like_dish(tail):
        rows.append((tail, previous_price))
    return rows


def infer_dish_fields(name: str, price: str, position: str) -> dict[str, object]:
    spicy_words = ["麻辣", "酸辣", "香辣", "辣", "尖椒", "咖喱", "麻"]
    soup_words = ["汤", "粥", "豆腐脑", "豆浆", "饮"]
    staple_words = ["饭", "面", "饺", "包", "饼", "馍", "烧麦", "馒头", "油条", "粉"]
    meat_words = ["肉", "鸡", "鸭", "鹅", "牛", "羊", "虾", "鱼", "排骨", "培根", "腊肠", "鸡蛋"]
    fried_words = ["炸", "煎", "炒", "烧饼", "手抓饼"]

    if any(word in name for word in soup_words):
        dish_type = "汤品"
        calories, protein, fat = 180, 7, 5
    elif any(word in name for word in staple_words):
        dish_type = "主食"
        calories, protein, fat = 560, 18, 18
    elif any(word in name for word in meat_words):
        dish_type = "荤菜"
        calories, protein, fat = 520, 28, 26
    else:
        dish_type = "素菜"
        calories, protein, fat = 260, 8, 12

    if any(word in name for word in fried_words):
        calories += 100
        fat += 8
    if "虾" in name or "牛" in name or "鸡" in name or "肉" in name:
        protein += 8
    if "粥" in name or "豆浆" in name or "茶叶蛋" in name:
        calories = min(calories, 220)

    spicy = 2 if any(word in name for word in spicy_words) else 0
    taste = "香辣" if spicy else "咸香"
    if "酸" in name:
        taste = "酸辣" if spicy else "酸甜"
    if "豆浆" in name or "粥" in name:
        taste = "清淡"

    tags = ["中餐", dish_type]
    if "早餐" in name or any(word in name for word in ["包", "饼", "粥", "豆浆", "茶叶蛋", "烧麦"]):
        tags.append("早餐")
    if calories <= 260:
        tags.append("低热量")
    if protein >= 25:
        tags.append("高蛋白")
    if fat >= 30:
        tags.append("高脂")
    if spicy:
        tags.append("重口味")
    if dish_type == "主食":
        tags.append("主食")

    return {
        "name": name,
        "calories": calories,
        "protein": protein,
        "fat": fat,
        "price": price,
        "spicy": spicy,
        "type": dish_type,
        "taste": taste,
        "tags": ";".join(dict.fromkeys(tags)),
        "position": position,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch OCR canteen menu images.")
    parser.add_argument("--input", required=True, type=Path, help="图片文件或图片文件夹")
    parser.add_argument("--out-dir", default=Path("backend/data/ocr_output"), type=Path)
    parser.add_argument("--position", default="风味食堂二楼")
    parser.add_argument("--min-score", default=0.45, type=float)
    args = parser.parse_args()

    images = iter_images(args.input)
    if not images:
        raise SystemExit(f"没有找到图片：{args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "menu_ocr_raw.csv"
    candidates_path = args.out_dir / "menu_dish_candidates.csv"
    text_path = args.out_dir / "menu_ocr_text.txt"

    ocr = make_ocr()
    raw_rows: list[dict[str, object]] = []
    candidate_by_name: dict[str, dict[str, object]] = {}

    with text_path.open("w", encoding="utf-8") as text_file:
        for image in images:
            print(f"OCR: {image}")
            result = ocr.ocr(str(image), cls=True)
            rows = flatten_paddle_result(result)
            text_file.write(f"\n===== {image} =====\n")
            for line_index, (text, score) in enumerate(rows):
                text = clean_text(text)
                if not text or score < args.min_score:
                    continue
                text_file.write(f"{text}\t{score:.3f}\n")
                raw_rows.append(
                    {
                        "image": str(image),
                        "line_index": line_index,
                        "text": text,
                        "score": round(score, 4),
                    }
                )
                for name, price in split_menu_line(text):
                    if not looks_like_dish(name):
                        continue
                    row = infer_dish_fields(name, price, args.position)
                    candidate_by_name.setdefault(name, row)

    with raw_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "line_index", "text", "score"])
        writer.writeheader()
        writer.writerows(raw_rows)

    fieldnames = ["name", "calories", "protein", "fat", "price", "spicy", "type", "taste", "tags", "position"]
    with candidates_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_by_name.values())

    print(f"图片数：{len(images)}")
    print(f"OCR 原文：{raw_path}")
    print(f"纯文本：{text_path}")
    print(f"候选菜品 CSV：{candidates_path}")
    print("提示：候选菜品需要人工校对，尤其是模糊图片、竖排菜单和价格。")


if __name__ == "__main__":
    main()
