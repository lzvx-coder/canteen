import csv
import json
import random
import re
import socket
import threading
import webbrowser
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

try:
    from .ml_inference import MODEL_PATH, predict_food_image
except ImportError:
    from ml_inference import MODEL_PATH, predict_food_image


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = BASE_DIR / "data" / "dishes.csv"
FRONTEND_INDEX = PROJECT_DIR / "frontend" / "index.html"

Goal = Literal["减脂", "增肌", "普通", "控预算"]


@dataclass
class Dish:
    name: str
    calories: float
    protein: float
    fat: float
    price: float
    spicy: int
    type: str
    taste: str
    tags: list[str]
    position: str = ""


@dataclass
class RecommendRequest:
    budget: float = 20
    goal: Goal = "减脂"
    taste: str = "清淡"
    preferences: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    top_k: int = 3


@dataclass
class PackageScore:
    dishes: list[Dish]
    total: dict[str, float]
    score: float
    nutrition_score: float
    taste_score: float
    budget_score: float
    preference_score: float
    health_score: float
    diversity_score: float
    reason: str


@dataclass
class RecognitionCandidate:
    dish: Dish
    confidence: float
    matched_features: list[str]
    explanation: str


def load_dishes() -> list[Dish]:
    def parse_number(value: str, default: float = 0.0) -> float:
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else default

    def parse_spicy(value: str) -> int:
        text = str(value).strip()
        if text in {"是", "辣", "微辣"}:
            return 1
        if text in {"中辣"}:
            return 2
        if text in {"重辣", "麻辣", "特辣"}:
            return 3
        if text in {"否", "不辣", "无", ""}:
            return 0
        return int(parse_number(text, 0))

    dishes: list[Dish] = []
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dishes.append(
                Dish(
                    name=row["name"],
                    calories=parse_number(row["calories"]),
                    protein=parse_number(row["protein"]),
                    fat=parse_number(row["fat"]),
                    price=parse_number(row["price"]),
                    spicy=parse_spicy(row["spicy"]),
                    type=row["type"],
                    taste=row["taste"],
                    tags=[tag.strip() for tag in row["tags"].split(";") if tag.strip()],
                    position=row.get("position", ""),
                )
            )
    return dishes


DISHES = load_dishes()

GOAL_OPTIONS = [
    {"value": "减脂", "label": "减脂", "description": "优先控制热量和脂肪，同时保证蛋白质。"},
    {"value": "增肌", "label": "增肌", "description": "优先高蛋白和足量能量，适合训练后正餐。"},
    {"value": "普通", "label": "普通", "description": "热量、蛋白质、价格相对均衡。"},
    {"value": "控预算", "label": "控预算", "description": "优先不超预算，再兼顾营养和口味。"},
]

CORE_PREFERENCE_TAGS = ["盖饭", "面条", "凉面", "卤肉饭", "汉堡", "炸鸡", "咖啡", "牛奶", "果茶", "乳茶"]
GENERIC_TAGS = {"中餐", "主食", "饮品", "饮料", "荤菜", "素菜", "小吃", "低热量", "普通能量", "高蛋白", "高脂", "高热量", "自选"}


def menu_options() -> dict[str, object]:
    tag_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    taste_counts: dict[str, int] = {}
    position_counts: dict[str, int] = {}
    for dish in DISHES:
        if dish.type:
            type_counts[dish.type] = type_counts.get(dish.type, 0) + 1
        if dish.position:
            position_counts[dish.position] = position_counts.get(dish.position, 0) + 1
        for taste in re.split(r"[;；、,\s]+", dish.taste):
            if taste:
                taste_counts[taste] = taste_counts.get(taste, 0) + 1
        for tag in dish.tags:
            if tag and tag not in GENERIC_TAGS:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    preference_values = []
    for value in CORE_PREFERENCE_TAGS:
        if value not in preference_values:
            preference_values.append(value)
    ranked_tags = sorted(tag_counts, key=lambda item: (-tag_counts[item], item))
    ranked_types = sorted(type_counts, key=lambda item: (-type_counts[item], item))
    for value in [*ranked_tags, *ranked_types]:
        if value not in preference_values and len(preference_values) < 36:
            preference_values.append(value)

    avoid_values = [
        "辣",
        "麻辣",
        "牛肉",
        "猪肉",
        "鸡肉",
        "鱼虾海鲜",
        "鸡蛋",
        "奶制品",
        "咖啡",
        "油炸",
        "高脂",
        "高热量",
        "甜",
        "冷饮",
    ]
    available_avoid = []
    all_text = " ".join(
        f"{dish.name} {dish.type} {dish.taste} {' '.join(dish.tags)}"
        for dish in DISHES
    )
    for value in avoid_values:
        terms = preference_terms(value)
        if any(term in all_text for term in terms) or value in {"辣", "甜"}:
            available_avoid.append(value)

    return {
        "goals": GOAL_OPTIONS,
        "preferences": [{"value": item, "count": tag_counts.get(item, type_counts.get(item, 0))} for item in preference_values],
        "avoid": [{"value": item} for item in available_avoid],
        "positions": [{"value": item, "count": position_counts[item]} for item in sorted(position_counts)],
        "tastes": [{"value": item, "count": taste_counts[item]} for item in sorted(taste_counts, key=lambda name: (-taste_counts[name], name))[:24]],
    }

INGREDIENT_KEYWORDS = {
    "鸡": ["鸡", "鸡胸", "鸡丁", "鸡排", "鸡腿", "鸡块"],
    "牛": ["牛", "牛肉", "牛柳", "牛腩"],
    "鱼": ["鱼", "鱼块", "鱼香"],
    "虾": ["虾", "虾仁"],
    "蛋": ["蛋", "鸡蛋", "滑蛋"],
    "豆腐": ["豆腐", "麻婆", "香煎"],
    "米饭": ["米饭", "糙米", "饭"],
    "面": ["面", "凉面"],
    "土豆": ["土豆"],
    "番茄": ["番茄", "西红柿"],
    "西兰花": ["西兰花"],
    "青椒": ["青椒"],
    "黄瓜": ["黄瓜"],
    "玉米": ["玉米"],
    "南瓜": ["南瓜"],
    "紫菜": ["紫菜"],
    "冬瓜": ["冬瓜"],
    "排骨": ["排骨"],
    "生菜": ["生菜"],
    "水果": ["水果"],
    "酸奶": ["酸奶"],
}

VISUAL_KEYWORDS = {
    "绿色": ["西兰花", "青椒", "青菜", "生菜", "黄瓜", "蔬菜"],
    "红色": ["番茄", "辣子", "红烧", "咖喱"],
    "白色": ["米饭", "豆腐", "鱼", "酸奶"],
    "黄色": ["鸡蛋", "玉米", "南瓜", "土豆", "咖喱"],
    "汤": ["汤", "粥"],
    "主食": ["米饭", "面", "粥", "玉米"],
    "肉类": ["鸡", "牛", "肉", "排骨"],
    "素菜": ["素菜", "蔬菜", "豆腐", "生菜", "黄瓜"],
    "辣": ["麻辣", "辣", "水煮", "宫保", "辣子"],
    "清淡": ["清淡", "清蒸", "凉拌", "水煮", "汤"],
}


def sum_package(dishes: tuple[Dish, ...]) -> dict[str, float]:
    return {
        "calories": round(sum(d.calories for d in dishes), 1),
        "protein": round(sum(d.protein for d in dishes), 1),
        "fat": round(sum(d.fat for d in dishes), 1),
        "price": round(sum(d.price for d in dishes), 1),
    }


def target_ranges(goal: Goal) -> dict[str, tuple[float, float]]:
    ranges = {
        "减脂": {"calories": (350, 620), "protein": (25, 60), "fat": (5, 22)},
        "增肌": {"calories": (620, 950), "protein": (35, 85), "fat": (10, 35)},
        "控预算": {"calories": (450, 800), "protein": (18, 60), "fat": (5, 32)},
        "普通": {"calories": (450, 780), "protein": (20, 65), "fat": (8, 30)},
    }
    return ranges[goal]


def range_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    width = max(high - low, 1)
    center = (low + high) / 2
    distance = abs(value - center) - width / 2
    return max(0.0, 1 - distance / width)


def nutrition_score(total: dict[str, float], goal: Goal) -> float:
    ranges = target_ranges(goal)
    cal = range_score(total["calories"], *ranges["calories"])
    protein = range_score(total["protein"], *ranges["protein"])
    fat = range_score(total["fat"], *ranges["fat"])
    if goal in {"减脂", "增肌"}:
        return 0.45 * protein + 0.35 * cal + 0.20 * fat
    return 0.34 * protein + 0.33 * cal + 0.33 * fat


def taste_score(dishes: tuple[Dish, ...], preferred_taste: str, avoid: list[str]) -> float:
    avoid_text = " ".join(avoid)
    avoid_terms = expanded_terms(avoid)
    matched = sum(1 for dish in dishes if preferred_taste in {dish.taste, *dish.tags})
    spicy_limit = 1 if "辣" in avoid_text else 3
    spicy_penalty = sum(max(0, dish.spicy - spicy_limit) for dish in dishes) * 0.18
    avoid_penalty = 0.0
    for dish in dishes:
        search_text = f"{dish.name} {dish.type} {dish.taste} {' '.join(dish.tags)}"
        if any(word and word in search_text for word in avoid_terms):
            avoid_penalty += 0.35
    return max(0.0, min(1.0, matched / len(dishes) + 0.25 - spicy_penalty - avoid_penalty))


def budget_score(price: float, budget: float) -> float:
    if price <= budget:
        return 1 - max(0, budget - price) / max(budget, 1) * 0.25
    return max(0.0, 1 - (price - budget) / max(budget, 1) * 1.5)


PREFERENCE_ALIASES = {
    "乳茶": ["乳茶", "奶茶", "牛乳茶", "轻乳茶"],
    "牛奶": ["牛奶", "牛乳", "奶昔", "拿铁", "奶香"],
    "凉面": ["凉面", "冷面", "拌面"],
    "卤肉饭": ["卤肉饭", "卤肉", "肉燥饭"],
    "汉堡": ["汉堡", "堡"],
    "炸鸡": ["炸鸡", "鸡排", "鸡块", "鸡腿"],
    "牛肉": ["牛肉", "牛柳", "牛腩", "肥牛"],
    "猪肉": ["猪肉", "肉丝", "肉片", "肉末", "回锅肉", "红烧肉", "卤肉", "排骨", "肥肠"],
    "鸡肉": ["鸡肉", "鸡丁", "鸡排", "鸡腿", "鸡块", "鸡柳"],
    "鱼虾海鲜": ["鱼", "虾", "海鲜", "鱼香", "烤鱼"],
    "鸡蛋": ["鸡蛋", "蛋", "滑蛋", "煎蛋", "卤蛋"],
    "奶制品": ["牛奶", "牛乳", "奶茶", "乳茶", "奶昔", "酸奶", "拿铁", "奶香"],
    "油炸": ["油炸", "炸鸡", "鸡排", "脆皮", "薯条", "炸"],
    "甜": ["甜", "酸甜", "清甜", "奶茶", "果茶", "甜品"],
    "冷饮": ["冷", "冰", "冰沙", "冷饮"],
}


def preference_terms(preference: str) -> list[str]:
    return PREFERENCE_ALIASES.get(preference, [preference])


def expanded_terms(items: list[str]) -> list[str]:
    return normalize_features([term for item in items for term in preference_terms(item)])


def preference_score(dishes: tuple[Dish, ...], preferences: list[str]) -> float:
    selected = [item for item in preferences if item and item != "不限"]
    if not selected:
        return 1.0
    terms = expanded_terms(selected)
    score = 0.0
    for dish in dishes:
        searchable = f"{dish.name} {dish.type} {dish.taste} {' '.join(dish.tags)}"
        if any(term in searchable for term in terms):
            score += 1.0
    return min(1.0, score / len(dishes))


def health_score(dishes: tuple[Dish, ...], total: dict[str, float], goal: Goal) -> float:
    low_fat_bonus = 0.12 if total["fat"] <= target_ranges(goal)["fat"][1] else -0.1
    vegetable_bonus = 0.12 if any(d.type in {"素菜", "荤素搭配", "汤品"} for d in dishes) else 0
    protein_bonus = 0.14 if total["protein"] >= target_ranges(goal)["protein"][0] else 0
    spicy_penalty = 0.08 * sum(1 for d in dishes if d.spicy >= 3)
    fried_penalty = 0.12 * sum(1 for d in dishes if "辣子" in d.name or "煎" in d.name)
    return max(0.0, min(1.0, 0.62 + low_fat_bonus + vegetable_bonus + protein_bonus - spicy_penalty - fried_penalty))


def diversity_score(dishes: tuple[Dish, ...], history: list[str]) -> float:
    if not history:
        return 1.0
    repeated = sum(1 for dish in dishes if dish.name in history)
    return max(0.0, 1 - repeated / len(dishes) * 0.7)


def make_reason(goal: Goal, total: dict[str, float], names: list[str]) -> str:
    dish_text = " + ".join(names)
    if len(names) == 1:
        prefix = f"推荐单品 {dish_text}"
    else:
        prefix = f"推荐 {dish_text}，同一窗口加料搭配"
    if goal == "减脂":
        return f"{prefix}；热量约 {total['calories']} kcal，蛋白质 {total['protein']} g，脂肪控制较稳。"
    if goal == "增肌":
        return f"{prefix}；蛋白质达到 {total['protein']} g，并提供 {total['calories']} kcal 能量。"
    if goal == "控预算":
        return f"{prefix}；总价 {total['price']} 元，适合预算优先的学生。"
    return f"{prefix}；总热量 {total['calories']} kcal，适合作为日常午晚餐。"


def is_addon(dish: Dish) -> bool:
    text = f"{dish.name} {dish.type} {' '.join(dish.tags)}"
    return "加料" in text or dish.type in {"加料", "配菜"}


def same_position(dishes: tuple[Dish, ...]) -> bool:
    positions = {dish.position for dish in dishes if dish.position}
    return len(positions) <= 1


def passes_hard_filters(dishes: tuple[Dish, ...], request: RecommendRequest) -> bool:
    total = sum_package(dishes)
    price_limit = request.budget if request.goal == "控预算" else request.budget * 1.25
    if total["price"] > price_limit:
        return False
    if not same_position(dishes):
        return False
    if len(dishes) > 1 and sum(1 for dish in dishes if not is_addon(dish)) != 1:
        return False
    if len(dishes) == 1 and is_addon(dishes[0]):
        return False
    avoid_text = " ".join(request.avoid)
    if "辣" in avoid_text and any(d.spicy >= 3 for d in dishes):
        return False
    names = [d.name for d in dishes]
    return len(names) == len(set(names))


def rank_packages(request: RecommendRequest) -> list[PackageScore]:
    packages: list[PackageScore] = []
    selected_positions = {position for position in request.positions if position and position != "不限"}
    avoid_terms = expanded_terms(request.avoid)

    def matches_request_preferences(dish: Dish) -> bool:
        if not request.preferences:
            return True
        text = f"{dish.name} {dish.type} {dish.taste} {' '.join(dish.tags)}"
        if any(pref in {"绿色", "素菜"} for pref in request.preferences):
            if any(term in text for term in ["鸡", "鸭", "鱼", "虾", "牛", "猪", "肉", "排", "肥牛", "饭", "炒饭", "盖饭"]):
                return False
            return any(term in text for term in ["素菜", "蔬菜", "清淡", "低热量", "豆腐", "青菜", "生菜", "娃娃菜", "菌菇", "土豆", "时蔬", "自选"])
        terms = [term for preference in request.preferences for term in preference_terms(preference)]
        if terms:
            return any(term in text for term in terms)
        return True

    base_candidates = [
        dish
        for dish in DISHES
        if dish.price <= request.budget * 1.25
        and (not selected_positions or dish.position in selected_positions)
        and dish.name not in request.history
        and not any(word and word in f"{dish.name} {dish.type} {dish.taste} {' '.join(dish.tags)}" for word in avoid_terms)
    ]
    preferred_candidates = [dish for dish in base_candidates if matches_request_preferences(dish)]
    if preferred_candidates:
        base_candidates = preferred_candidates
    if not base_candidates:
        base_candidates = [
            dish
            for dish in DISHES
            if dish.price <= request.budget * 1.25
            and (not selected_positions or dish.position in selected_positions)
        ] or list(DISHES)

    mains = sorted(
        [dish for dish in base_candidates if not is_addon(dish)],
        key=lambda dish: (
            dish.price <= request.budget,
            dish.protein,
            -dish.fat,
            -dish.price,
        ),
        reverse=True,
    )[:120]
    addons = sorted(
        [dish for dish in base_candidates if is_addon(dish)],
        key=lambda dish: (dish.protein, -dish.fat, -dish.price),
        reverse=True,
    )

    combos: list[tuple[Dish, ...]] = [(dish,) for dish in mains]
    for main in mains:
        same_window_addons = [addon for addon in addons if addon.position == main.position][:12]
        combos.extend((main, addon) for addon in same_window_addons)

    for combo in combos:
        if not passes_hard_filters(combo, request):
            continue
        total = sum_package(combo)
        n_score = nutrition_score(total, request.goal)
        t_score = taste_score(combo, request.taste, request.avoid)
        b_score = budget_score(total["price"], request.budget)
        p_score = preference_score(combo, request.preferences)
        h_score = health_score(combo, total, request.goal)
        d_score = diversity_score(combo, request.history)
        combo_penalty = 0.08 * max(0, len(combo) - 1)
        score = (
            0.31 * n_score
            + 0.21 * t_score
            + 0.16 * b_score
            + 0.16 * p_score
            + 0.10 * h_score
            + 0.06 * d_score
            - combo_penalty
        )
        packages.append(
            PackageScore(
                dishes=list(combo),
                total=total,
                score=round(score * 100, 2),
                nutrition_score=round(n_score * 100, 2),
                taste_score=round(t_score * 100, 2),
                budget_score=round(b_score * 100, 2),
                preference_score=round(p_score * 100, 2),
                health_score=round(h_score * 100, 2),
                diversity_score=round(d_score * 100, 2),
                reason=make_reason(request.goal, total, [dish.name for dish in combo]),
            )
        )
    singles = sorted((item for item in packages if len(item.dishes) == 1), key=lambda item: item.score, reverse=True)
    combos = sorted((item for item in packages if len(item.dishes) > 1), key=lambda item: item.score, reverse=True)

    def varied_pick(items: list[PackageScore], limit: int) -> list[PackageScore]:
        if limit <= 0 or not items:
            return []
        pool = items[: max(limit * 6, min(30, len(items)))]
        random.shuffle(pool)
        pool.sort(key=lambda item: item.score + random.uniform(-3.0, 3.0), reverse=True)
        return pool[:limit]

    selected = varied_pick(singles, request.top_k)
    if len(selected) < request.top_k:
        selected.extend(varied_pick(combos, request.top_k - len(selected)))
    return selected


def normalize_features(features: list[str]) -> list[str]:
    normalized: list[str] = []
    for feature in features:
        for token in re.split(r"[,，、\s]+", feature):
            token = token.strip()
            if token and token not in normalized:
                normalized.append(token)
    return normalized


FEATURE_ALIASES = {
    "绿色": ["绿", "青菜", "蔬菜", "素菜", "生菜", "白菜", "油麦菜", "娃娃菜", "菜花", "西兰花", "青椒", "笋", "清淡"],
    "红色": ["番茄", "西红柿", "辣", "红烧", "麻辣", "香辣"],
    "黄色": ["鸡蛋", "玉米", "土豆", "南瓜", "咖喱"],
    "白色": ["豆腐", "米饭", "鱼", "虾", "蛋白"],
    "深色": ["肉", "牛", "猪", "鸭", "酱", "卤", "红烧"],
    "浅色": ["清淡", "汤", "粥"],
    "素菜": ["素菜", "蔬菜", "青菜", "白菜", "娃娃菜", "菜花", "生菜", "土豆", "豆腐", "菌", "菇", "笋"],
    "肉类": ["荤菜", "肉", "牛", "猪", "鸡", "鸭", "鱼", "虾", "排骨", "肥牛"],
    "主食": ["主食", "米饭", "饭", "面", "粉", "饺子", "包子"],
    "米饭": ["米饭", "饭", "盖饭", "拌饭", "炒饭"],
    "面食": ["面食", "面", "饺子", "水饺", "包子"],
    "汤": ["汤", "汤品", "粥", "砂锅", "煲"],
    "豆腐": ["豆腐", "千叶豆腐", "鱼豆腐", "豆花", "腐竹", "千张"],
    "清淡": ["清淡", "清爽", "鲜美", "咸鲜", "清甜", "脆嫩"],
    "辣": ["辣", "麻辣", "香辣", "酸辣", "鲜辣", "微辣"],
    "鸡": ["鸡", "鸡腿", "鸡排", "鸡柳", "鸡块", "鸡丁"],
    "鸡蛋": ["鸡蛋", "蛋", "滑蛋", "蛋花", "煎蛋", "卤蛋"],
}


FEATURE_WEIGHTS = {
    "绿色": 1.5,
    "素菜": 1.7,
    "清淡": 1.0,
    "白色": 0.35,
    "米饭": 0.45,
    "主食": 0.35,
    "豆腐": 1.15,
    "汤": 1.0,
    "肉类": 1.45,
    "红色": 1.0,
    "黄色": 0.9,
    "辣": 1.0,
    "鸡": 1.2,
    "鸡蛋": 1.1,
}


def feature_terms(feature: str) -> list[str]:
    aliases = FEATURE_ALIASES.get(feature, [])
    legacy_aliases = INGREDIENT_KEYWORDS.get(feature, []) + VISUAL_KEYWORDS.get(feature, [])
    return normalize_features([feature, *aliases, *legacy_aliases])


def dish_matches_feature(dish: Dish, feature: str, searchable: str) -> bool:
    if feature == "素菜":
        return "素菜" in dish.tags or dish.type == "素菜"
    if feature == "肉类":
        return "荤菜" in dish.tags
    if feature == "主食":
        return "主食" in dish.tags
    if feature == "米饭":
        return "主食" in dish.tags and any(term in dish.name or term in dish.type for term in ["饭", "米饭", "盖饭", "拌饭", "炒饭"])
    if feature == "面食":
        return "面食" in dish.tags
    if feature == "汤":
        return "汤品" in dish.tags or "汤" in dish.type or "汤" in dish.name
    return any(term and term in searchable for term in feature_terms(feature))


def feature_conflict_penalty(dish: Dish, features: list[str]) -> float:
    feature_set = set(features)
    penalty = 0.0
    if {"绿色", "素菜"} & feature_set:
        if "荤菜" in dish.tags:
            penalty += 0.65 if "素菜" in dish.tags else 1.4
        if "主食" in dish.tags and "米饭" not in feature_set and "主食" not in feature_set:
            penalty += 0.45
        if dish.calories >= 650:
            penalty += 0.35
        if dish.fat >= 24:
            penalty += 0.25
    if "肉类" in feature_set and "荤菜" not in dish.tags:
        penalty += 0.9
    if "米饭" not in feature_set and "主食" not in feature_set and any(term in dish.name or term in dish.type for term in ["盖饭", "拌饭", "炒饭"]):
        penalty += 0.5
    if "辣" not in feature_set and any(term in dish.taste for term in ["麻辣", "香辣", "酸辣", "鲜辣", "特辣"]):
        penalty += 0.35
    return penalty


def is_vegetable_like(dish: Dish) -> bool:
    text = f"{dish.name} {dish.type} {dish.taste} {' '.join(dish.tags)}"
    veggie_terms = ["素菜", "蔬菜", "青菜", "白菜", "娃娃菜", "生菜", "菜花", "油麦", "菠菜", "时蔬", "笋", "豆腐", "低热量", "清淡"]
    return any(term in text for term in veggie_terms) and "饭" not in dish.name


def fits_green_vegetable_scene(dish: Dish, features: list[str]) -> bool:
    if not ({"绿色", "素菜"} & set(features)):
        return True
    if is_addon(dish):
        return False
    if not is_vegetable_like(dish):
        return False
    text = f"{dish.name} {dish.type} {dish.taste} {' '.join(dish.tags)}"
    if "辣" not in features and any(term in text for term in ["麻辣", "香辣", "酸辣", "辣"]):
        return False
    if "主食" not in features and any(term in text for term in ["主食", "面食", "面点", "米粉", "饭", "粥"]):
        return False
    return True


def recognition_sort_key(candidate: RecognitionCandidate, features: list[str]) -> tuple[float, float]:
    dish = candidate.dish
    priority = 0.0
    if {"绿色", "素菜"} & set(features):
        text = f"{dish.name} {dish.type} {' '.join(dish.tags)}"
        if dish.type == "素菜":
            priority += 0.35
        if any(term in text for term in ["青菜", "生菜", "白菜", "娃娃菜", "菜花", "时蔬", "笋", "菌菇", "杏鲍菇"]):
            priority += 0.25
        if any(term in text for term in ["豆腐脑", "粥", "面", "饭", "包", "饺"]):
            priority -= 0.35
    return (candidate.confidence, priority)


def extract_image_clues(image_name: str = "", image_data: str = "") -> list[str]:
    clues: list[str] = []
    text = image_name.lower()
    for keyword, synonyms in INGREDIENT_KEYWORDS.items():
        if keyword in image_name or any(item.lower() in text for item in synonyms):
            clues.append(keyword)
    for keyword, synonyms in VISUAL_KEYWORDS.items():
        if keyword in image_name or any(item.lower() in text for item in synonyms):
            clues.append(keyword)

    return normalize_features(clues)


def model_prediction_terms(public_predictions: list[dict[str, object]]) -> list[str]:
    terms: list[str] = []
    for prediction in public_predictions:
        dish_name = str(prediction.get("dish_name", "")).replace("_", " ").strip()
        label = str(prediction.get("label", "")).replace("_", " ").strip()
        confidence = float(prediction.get("confidence", 0))
        if confidence < 0.08:
            continue
        terms.extend(normalize_features([dish_name, label]))
        for token in normalize_features([dish_name, label]):
            terms.extend(preference_terms(token))
    return normalize_features(terms)


def model_match_score(dish: Dish, terms: list[str]) -> tuple[float, list[str]]:
    if not terms:
        return 0.0, []
    searchable = f"{dish.name} {dish.type} {dish.taste} {' '.join(dish.tags)}"
    matched = [term for term in terms if term and term in searchable]
    if not matched:
        return 0.0, []
    exact_bonus = 1.2 if any(term == dish.name for term in matched) else 0.0
    name_bonus = 0.8 if any(term in dish.name for term in matched) else 0.0
    return min(3.0, 0.55 * len(set(matched)) + exact_bonus + name_bonus), matched[:5]


def public_recognition_result(public_predictions: list[dict[str, object]], features: list[str]) -> dict[str, object]:
    if public_predictions:
        best = public_predictions[0]
        label = str(best.get("label", ""))
        dish_name = str(best.get("dish_name", ""))
        label_text = label.replace("_", " ").strip()
        name = dish_name if dish_name and dish_name != label else label_text
        return {
            "source": "model",
            "name": name,
            "label": label,
            "confidence": float(best.get("confidence", 0)),
            "note": "模型识别结果，作为图片本身的优先判断。",
        }
    if features:
        return {
            "source": "visual_rules",
            "name": " / ".join(features[:4]),
            "label": "",
            "confidence": 0,
            "note": "未检测到可用模型结果，使用图片颜色和手动线索辅助判断。",
        }
    return {
        "source": "none",
        "name": "未识别",
        "label": "",
        "confidence": 0,
        "note": "请上传图片，或补充识别线索。",
    }


def recognize_dish(data: dict[str, object]) -> dict[str, object]:
    manual_features = normalize_features([str(item) for item in data.get("features", [])])
    browser_features = normalize_features([str(item) for item in data.get("image_features", [])])
    image_clues = extract_image_clues(str(data.get("image_name", "")), str(data.get("image_data", "")))
    features = normalize_features([*manual_features, *browser_features, *image_clues])
    if "绿色" in features and "素菜" in features:
        features = [feature for feature in features if feature not in {"白色", "米饭", "主食"}]
    model_predictions = predict_food_image(str(data.get("image_data", "")))
    public_predictions = [
        {
            "label": str(prediction.get("label", "")),
            "dish_name": str(prediction.get("dish_name", "")),
            "confidence": float(prediction.get("confidence", 0)),
        }
        for prediction in model_predictions
    ]
    model_terms = model_prediction_terms(public_predictions)
    recognition_result = public_recognition_result(public_predictions, features)

    candidates: list[RecognitionCandidate] = []

    for dish in DISHES:
        if not fits_green_vegetable_scene(dish, features):
            continue
        searchable = f"{dish.name} {dish.type} {dish.taste} {' '.join(dish.tags)}"
        matched: list[str] = []
        raw_score = 0.0
        model_score_value, model_matched = model_match_score(dish, model_terms)
        for feature in features:
            if dish_matches_feature(dish, feature, searchable):
                matched.append(feature)
                raw_score += FEATURE_WEIGHTS.get(feature, 1.0)
        if raw_score > 0 or model_score_value > 0:
            matched = normalize_features([*model_matched, *matched])
            conflict_penalty = feature_conflict_penalty(dish, features)
            nutrition_bonus = 0.06 if dish.protein >= 25 and "肉类" in features else 0
            vegetable_bonus = 0.22 if {"绿色", "素菜"} & set(features) and dish.type == "素菜" else 0
            normalized_score = max(0.0, raw_score - conflict_penalty) / max(len(features), 1)
            model_bonus = min(0.48, model_score_value * 0.16)
            confidence = min(0.98, 0.30 + normalized_score * 0.42 + model_bonus + nutrition_bonus + vegetable_bonus)
            candidates.append(
                RecognitionCandidate(
                    dish=dish,
                    confidence=round(confidence, 2),
                    matched_features=matched,
                    explanation=f"优先结合模型预测，再匹配到 {', '.join(matched)}，并按学校菜品库排序。",
                )
            )

    if not candidates:
        fallback_source = [dish for dish in DISHES if fits_green_vegetable_scene(dish, features)] if {"绿色", "素菜"} & set(features) else list(DISHES)
        fallback = sorted(fallback_source, key=lambda dish: (dish.calories <= 350, -dish.fat, dish.protein), reverse=True)[:5]
        candidates = [
            RecognitionCandidate(
                dish=dish,
                confidence=0.28,
                matched_features=[],
                explanation="未识别到明确线索，返回高蛋白且相对健康的候选菜品。",
            )
            for dish in fallback
        ]

    candidates.sort(key=lambda item: recognition_sort_key(item, features), reverse=True)
    best = candidates[0]
    public_best = public_predictions[0] if public_predictions else None
    request = RecommendRequest(
        budget=float(data.get("budget", 22)),
        goal=str(data.get("goal", "普通")) if str(data.get("goal", "普通")) in {"减脂", "增肌", "普通", "控预算"} else "普通",
        taste=best.dish.taste,
        preferences=normalize_features(
            [
                *[str(item) for item in data.get("preferences", []) if str(item).strip() and str(item).strip() != "不限"],
                *features,
            ]
        ),
        positions=[str(item) for item in data.get("positions", []) if str(item).strip() and str(item).strip() != "不限"],
        avoid=[str(item) for item in data.get("avoid", []) if str(item).strip()],
        history=[str(item) for item in data.get("history", []) if str(item).strip()],
        top_k=3,
    )
    return {
        "features": features,
        "model_enabled": MODEL_PATH.exists(),
        "model_predictions": model_predictions,
        "public_predictions": public_predictions,
        "public_best": public_best,
        "recognition_result": recognition_result,
        "model_terms": model_terms,
        "best_match": best,
        "candidates": candidates[:5],
        "followup_recommendations": rank_packages(request),
    }


def health() -> dict[str, str]:
    return {"status": "ok", "system": "智能食堂菜品推荐与营养分析系统"}


def dishes() -> list[Dish]:
    return DISHES


def options() -> dict[str, object]:
    return menu_options()


def recommend(request: RecommendRequest) -> dict[str, object]:
    return {
        "query": asdict(request),
        "weights": {
            "nutrition": 0.31,
            "taste": 0.21,
            "budget": 0.16,
            "preference": 0.16,
            "health": 0.10,
            "diversity": 0.06,
        },
        "recommendations": rank_packages(request),
    }


def demo_profiles() -> dict[str, object]:
    samples = [
        RecommendRequest(budget=20, goal="减脂", taste="清淡", avoid=["辣"], history=["宫保鸡丁"]),
        RecommendRequest(budget=28, goal="增肌", taste="咸香", avoid=[], history=["鸡腿饭"]),
        RecommendRequest(budget=14, goal="控预算", taste="家常", avoid=["太辣"], history=["麻婆豆腐"]),
    ]
    names = ["减脂学生", "增肌学生", "控预算学生"]
    return {
        "profiles": [
            {
                "name": name,
                "input": asdict(sample),
                "top_recommendation": rank_packages(sample)[0],
            }
            for name, sample in zip(names, samples)
        ]
    }


def request_from_dict(data: dict[str, object]) -> RecommendRequest:
    goal = str(data.get("goal", "减脂"))
    if goal not in {"减脂", "增肌", "普通", "控预算"}:
        goal = "减脂"
    return RecommendRequest(
        budget=max(5, min(80, float(data.get("budget", 20)))),
        goal=goal,  # type: ignore[arg-type]
        taste=str(data.get("taste", "清淡")),
        preferences=[str(item) for item in data.get("preferences", []) if str(item).strip() and str(item).strip() != "不限"],
        positions=[str(item) for item in data.get("positions", []) if str(item).strip() and str(item).strip() != "不限"],
        avoid=[str(item) for item in data.get("avoid", []) if str(item).strip()],
        history=[str(item) for item in data.get("history", []) if str(item).strip()],
        top_k=max(1, min(8, int(data.get("top_k", 3)))),
    )


def to_json(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, default=asdict).encode("utf-8")


class CanteenHandler(BaseHTTPRequestHandler):
    server_version = "CanteenHTTP/1.0"

    def send_json(self, status: int, data: object) -> None:
        payload = to_json(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_html(self, status: int, path: Path) -> None:
        if not path.exists():
            self.send_json(404, {"error": "前端页面不存在"})
            return
        payload = path.read_bytes()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_json(200, {"status": "ok"})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html", "/frontend/index.html"}:
            self.send_html(200, FRONTEND_INDEX)
        elif path == "/health":
            self.send_json(200, health())
        elif path == "/dishes":
            self.send_json(200, dishes())
        elif path == "/options":
            self.send_json(200, options())
        elif path == "/profiles":
            self.send_json(200, demo_profiles())
        else:
            self.send_json(404, {"error": "接口不存在"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            if path == "/shutdown":
                self.send_json(200, {"status": "shutting_down"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            data = json.loads(raw)
            if path == "/recommend":
                self.send_json(200, recommend(request_from_dict(data)))
            elif path == "/recognize":
                self.send_json(200, recognize_dish(data))
            else:
                self.send_json(404, {"error": "接口不存在"})
        except Exception as exc:
            self.send_json(400, {"error": f"请求参数错误: {exc}"})

    def log_message(self, format: str, *args: object) -> None:
        return


class CanteenServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_server(host: str, port: int) -> CanteenServer:
    try:
        return CanteenServer((host, port), CanteenHandler)
    except OSError as exc:
        fallback_ports = [*range(port + 1, port + 21), 0]
        for candidate in fallback_ports:
            try:
                server = CanteenServer((host, candidate), CanteenHandler)
                actual_port = server.server_address[1]
                print(f"端口 {port} 不可用，已自动切换到 {actual_port}。原错误: {exc}")
                return server
            except OSError:
                continue
        raise


def local_ip_addresses() -> list[str]:
    addresses: list[str] = []
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = item[4][0]
            if address.startswith("127.") or address in addresses:
                continue
            addresses.append(address)
    except OSError:
        pass
    return addresses


def run(host: str = "127.0.0.1", port: int = 8004, open_browser: bool = True, auto_shutdown: bool = True) -> None:
    server = make_server(host, port)
    actual_host, actual_port = server.server_address[:2]
    display_host = "127.0.0.1" if actual_host in {"0.0.0.0", ""} else actual_host
    query = "?auto_shutdown=1" if auto_shutdown else ""
    url = f"http://{display_host}:{actual_port}/{query}"
    print(f"智能食堂推荐系统已启动: {url}")
    if actual_host in {"0.0.0.0", ""}:
        for address in local_ip_addresses():
            print(f"局域网访问地址: http://{address}:{actual_port}/")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
