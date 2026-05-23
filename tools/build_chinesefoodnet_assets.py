from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "backend" / "data"
LABELS_PATH = DATA_DIR / "chinesefoodnet_labels.csv"
DISHES_PATH = DATA_DIR / "dishes.csv"


RAW_LABELS = """
0	麻婆豆腐	Mapo Tofu
1	家常豆腐	Home style sauteed Tofu
2	煎豆腐	Fried Tofu
3	豆腐花	Bean curd
4	臭豆腐	Stinky tofu
5	酸辣土豆丝	Potato silk
6	土豆泥	Pan fried potato
7	香煎土豆	Pan fried potato
8	土豆焖豆角	Braised beans with potato
9	地三鲜	Fried Potato, Green Pepper & Eggplant
10	薯条	French fries
11	鱼香茄子	Yu-Shiang Eggplant
12	蒜泥茄子	Mashed garlic eggplant
13	肉末茄子	Eggplant with mince pork
14	辣白菜	Spicy cabbage
15	醋溜白菜	Sour cabbage
16	上汤娃娃菜	Steamed Baby Cabbage
17	手撕包菜	Shredded cabbage
18	蚝油生菜	Sauteed Lettuce in Oyster Sauce
19	炒青菜	Saute vegetable
20	炒空心菜	tumis kangkung
21	蒜蓉油麦菜	Lettuce with smashed garlic
22	清炒菠菜	Sauteed spainch
23	炒豆芽	Sauteed bean sprouts
24	炒蚕豆	Sauteed broad beans
25	毛豆	Soybean
26	蚝油西兰花	Broccoli with Oyster Sauce
27	香煎藕盒	Deep Fried lotus root
28	莲藕	Lotus root
29	凉拌西红柿	Tomato salad
30	鸡鸭胗	Gizzard
31	凉拌木耳	Black Fungus in Vinegar Sauce
32	口水黄瓜	Cucumber in Sauce
33	花生米	peanut
34	凉拌海带丝	Seaweed salad
35	拔丝山药	Chinese Yam in Hot Toffee
36	清炒山药	Fried Yam
37	干煸豆角	Fried beans
38	蚝油杏鲍菇	Oyster mushroom
39	酿苦瓜	stuffed bitter melon
40	炒苦瓜	sauteed bitter melon
41	虎皮青椒	pepper with tiger skin
42	凉拌腐竹	Yuba salad
43	炒花菜	fried cauliflower
44	松仁玉米	Sauteed Sweet Corn with Pine Nuts
45	香菇青菜	Sauted Chinese Greens with Mushrooms
46	椒盐蘑菇	Spiced mushroom
47	芹菜香干	Celery and tofu
48	西芹百合	Sauteed Lily Bulbs and Celery
49	韭菜炒香干	Leak and tofu
50	西红柿炒鸡蛋	Scrambled egg with tomato
51	韭菜炒鸡蛋	Scrambled Egg with Leek
52	黄瓜炒鸡蛋	Scrambled Egg with cucumber
53	鸡蛋羹	Steamed egg custard
54	猪肝	Pork liver
55	猪耳朵	Pig ears
56	叉烧	roast pork
57	粉蒸排骨	Steamed pork with rice powder
58	糖醋排骨	Sweet and sour spareribs
59	海带炖排骨	Braised spareribs with kelp
60	可乐鸡翅	Cola Chicken wings
61	泡椒凤爪	Chicken Feet with Pickled Peppers
62	红烧鸡爪	Chicken Feet with black bean sauce
63	口水鸡	Steamed Chicken with Chili Sauce
64	烤鸭烧鹅	Roast goose
65	白斩鸡	Boiled chicken
66	大盘鸡	Saute Spicy Chicken
67	香菇蒸鸡	Steamed Chicken with Mushroom
68	黄焖鸡	chicken braised with brown sauce
69	豉油鸡	Soy sauce chicken
70	辣子鸡	Spicy Chicken
71	宫保鸡丁	Kung Pao Chicken
72	三杯鸡	Stewed Chicken with Three Cups Sauce
73	鸡丝、鸡丝面	Shredded chicken
74	炸鸡腿	Fried chicken drumsticks
75	啤酒鸭	Beer duck
76	腰花	Scalloped pork or lamb kidneys
77	红烧肉	Braised pork
78	红烧牛肉	Braised beef
79	酱牛肉	Beef Seasoned with Soy Sauce
80	西红柿牛腩	Sirloin tomatoes
81	土豆炖牛腩	Stewed sirloin potatoes
82	杭椒牛柳	Sauteed Beef Fillet with Hot Green Pepper
83	梅菜扣肉	Pork with salted vegetable
84	回锅肉	Double cooked pork slices
85	猪肉炖粉条	Braised Pork with Vermicelli
86	水煮肉片	Boiled Shredded pork in chili oil
87	糖醋里脊	Fried Sweet and Sour Tenderloin
88	咕噜肉	Cripsy sweet & sour pork slices
89	锅包肉	Pot bag meat
90	农家小炒肉	Shredded Pork with Vegetables
91	培根金针菇卷	Tiger lily buds in Baconic
92	京酱肉丝	Sauteed Shredded Pork in Sweet Bean Sauce
93	豆角肉丝	Shredded pork with bean
94	酱焖猪蹄	Braised pig feet with soy sauce
95	肚丝	Tripe
96	青椒肉丝	Shredded pork and green pepper
97	鱼香肉丝	Yu-Shiang Shredded Pork
98	木耳炒肉丝	Braised Fungus with pork slice
99	木须肉	Sauteed Sliced Pork,Eggs and Black Fungus
100	莴笋肉丝	Lettuce shredded meat
101	蚂蚁上树	Sauteed Vermicelli with Spicy Minced Pork
102	孜然羊肉	Fried Lamb with Cumin
103	羊肉串	Lamb shashlik
104	葱爆羊肉	Sauteed Sliced Lamb with Scallion
105	红烧狮子头	Stewed Pork Ball in Brown Sauce
106	酸菜鱼	Boiled Fish with Picked Cabbage and Chili
107	烤鱼	grilled fish
108	糖醋鲤鱼	Sweet and sour fish
109	松鼠桂鱼	Sweet and Sour Mandarin Fish
110	红烧带鱼	Braised Hairtail in Brown Sauce
111	剁椒鱼头	Steamed Fish Head with Diced Hot Red Peppers
112	水煮鱼	Fish Filets in Hot Chili Oil
113	清蒸鲈鱼	Steamed Perch
114	芝士虾球	Cheese Shrimp Meat
115	虾仁西兰花	Shrimp broccoli
116	油焖大虾	Braised Shrimp in chili oil
117	香辣虾	Spicy shrimp
118	香辣小龙虾	Spicy crayfish
119	水晶虾饺	Shrimp Duplings
120	蒜茸粉丝蒸虾	Steamed shrimp with garlic and vermicelli
121	清炒虾仁	Sauteed Shrimp meat
122	皮皮虾	Pipi shrimp
123	扇贝	Scallop in Shell
124	生蚝	Oysters
125	鱿鱼	squid
126	鲍鱼	Abalone
127	螃蟹	Crab
128	甲鱼	Turtle
129	鳝鱼	eel
130	扬州炒饭	Yangzhou fried rice
131	蛋包饭	Omelette
132	小笼汤包	Steamed Bun Stuffed
133	烧麦	Steamed Pork Dumplings
134	家常早餐鸡蛋饼	egg omelet
135	土豆鸡蛋饼	Potato omelet
136	鸡蛋灌饼	Egg pie cake
137	卤蛋	Marinated Egg
138	荷包蛋	Poached Egg
139	葱花手抓饼	Pine cake with Diced Scallion
140	芝麻烧饼	Sesame seed cake
141	肉夹馍	Chinese hamburger
142	韭菜盒子	Leek box
143	南瓜紫薯馒头	steamed bun with purple potato and pumpkin
144	馒头	steamed bun
145	包子	Steamed stuffed bun
146	南瓜饼	Pumpkin pie
147	披萨	Pizza
148	油条	Deep-Fried Dough Sticks
149	炸酱面	sauteed noodles with minced meat
150	重庆酸辣粉	Chongqing Hot and Sour Rice Noodles
151	凉拌凉面	Cold noodles
152	西红柿鸡蛋面	Noodles with egg and tomato
153	肉酱意大利面	spaghetti with meat sauce
154	茄汁拌面	Noodles with tomato sauce
155	凉皮	Cold Rice Noodles
156	担担面	Sichuan noodles with peppery sauce
157	臊子面	Qishan noodles
158	炒面	fried noodles
159	饺子	Dumplings
160	玉米棒	Corn Cob
161	红烧牛肉面	Braised beef noodle
162	河粉	fried rice noodles
163	肠粉	Steamed vermicelli roll
164	鲜肉小馄饨	Pork wonton
165	煎饺	Fried Dumplings
166	汤圆	Tang-yuan
167	小米粥	Millet congee
168	红薯粥	Sweet potato porridge
169	海蛰	Jellyfish
170	皮蛋瘦肉粥	Minced Pork Congee with Preserved Egg
171	大米粥	Rice porridge
172	米饭	Rice
173	紫菜包饭	Laver rice
174	石锅饭	Stone pot of rice
175	乌鸡汤	Black bone chicken soup
176	鲫鱼豆腐汤	Crucian and Bean Curd Soup
177	疙瘩汤	Dough Drop and Assorted Vegetable Soup
178	酸辣汤	Hot and Sour Soup
179	萝卜排骨汤	Pork ribs soup with radish
180	西红柿鸡蛋汤	Tomato and Egg Soup
181	西湖牛肉羹	West Lake beef soup
182	莲藕排骨汤	Lotus Root and Rib soup
183	紫菜蛋花汤	Seaweed and Egg Soup
184	海带豆腐汤	Seaweed tofu soup
185	玉米排骨汤	Corn and sparerib soup
186	菠菜猪肝汤	Spinach and pork liver soup
187	罗宋汤	Borsch
188	银耳汤	White fungus soup
189	冬瓜汤	White gourd soup
190	酱汤	Miso soup
191	毛血旺	Duck Blood in Chili Sauce
192	夫妻肺片	Pork Lungs in Chili Sauce
193	麻辣香锅	Spicy pot
194	黄金如意肉卷	Golden meat rolls
195	蛋糕	Chiffon Cake
196	蛋挞	Egg Tart
197	面包	Bread
198	牛角包	Croissant
199	吐司	toast
200	饼干	Biscuits
201	曲奇饼干	cookies
202	苏打饼干	Soda biscuit
203	双皮奶	Double skin milk
204	冰激凌	ice cream
205	鸡蛋布丁	Egg pudding
206	冰糖雪梨	Sweet stewed snow pear
207	水果沙拉	Fruit salad
""".strip()


def parse_labels() -> list[dict[str, str]]:
    labels: list[dict[str, str]] = []
    for line in RAW_LABELS.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"Bad label row: {line}")
        index, zh_name, en_name = parts
        labels.append(
            {
                "id": f"{int(index):03d}",
                "name": zh_name.strip(),
                "english_name": en_name.strip(),
            }
        )
    return labels


def infer_profile(name: str) -> dict[str, object]:
    spicy_words = ["麻", "辣", "酸辣", "泡椒", "剁椒", "水煮", "香辣", "担担", "毛血旺", "麻辣"]
    sweet_words = ["糖醋", "拔丝", "蛋糕", "蛋挞", "布丁", "冰激凌", "双皮奶", "雪梨"]
    soup_words = ["汤", "羹"]
    staple_words = ["饭", "面", "馒头", "包", "饼", "饺", "粉", "粥", "米饭", "油条", "凉皮", "肠粉", "河粉", "吐司", "面包"]
    seafood_words = ["鱼", "虾", "蟹", "贝", "生蚝", "鲍", "鱿", "鳝", "甲鱼"]
    meat_words = ["猪", "肉", "牛", "羊", "鸡", "鸭", "鹅", "排骨", "叉烧", "肝", "肚", "胗", "蹄", "翅"]
    egg_words = ["蛋"]
    fried_words = ["炸", "煎", "薯条", "锅包", "椒盐"]
    veg_words = [
        "豆腐",
        "土豆",
        "茄子",
        "白菜",
        "青菜",
        "空心菜",
        "菠菜",
        "豆芽",
        "西兰花",
        "莲藕",
        "木耳",
        "黄瓜",
        "海带",
        "山药",
        "豆角",
        "杏鲍菇",
        "苦瓜",
        "青椒",
        "腐竹",
        "花菜",
        "玉米",
        "香菇",
        "蘑菇",
        "芹",
        "百合",
        "西红柿",
    ]

    if any(word in name for word in soup_words):
        dish_type = "汤品"
        calories, protein, fat, price = 180, 10, 7, 8
    elif any(word in name for word in staple_words):
        dish_type = "主食"
        calories, protein, fat, price = 520, 16, 18, 10
    elif any(word in name for word in seafood_words):
        dish_type = "荤菜"
        calories, protein, fat, price = 430, 32, 18, 18
    elif any(word in name for word in meat_words):
        dish_type = "荤菜"
        calories, protein, fat, price = 560, 30, 30, 16
    elif any(word in name for word in egg_words):
        dish_type = "荤素搭配"
        calories, protein, fat, price = 360, 18, 20, 9
    elif any(word in name for word in veg_words):
        dish_type = "素菜"
        calories, protein, fat, price = 230, 8, 12, 7
    else:
        dish_type = "小吃"
        calories, protein, fat, price = 360, 10, 16, 9

    if any(word in name for word in fried_words):
        calories += 120
        fat += 12
    if "红烧" in name or "焖" in name or "炖" in name:
        calories += 80
        fat += 6
    if any(word in name for word in sweet_words):
        taste = "甜香"
        spicy = 0
    elif any(word in name for word in spicy_words):
        taste = "香辣"
        spicy = 2 if "麻辣" not in name and "水煮" not in name else 3
    elif "凉拌" in name or "口水" in name:
        taste = "爽口"
        spicy = 1
    elif "酸" in name or "鱼香" in name:
        taste = "酸甜"
        spicy = 1
    else:
        taste = "咸香"
        spicy = 0

    tags = ["中餐", dish_type]
    if protein >= 25:
        tags.append("高蛋白")
    if fat >= 30:
        tags.append("高脂")
    if calories <= 260:
        tags.append("低热量")
    if spicy >= 2:
        tags.append("重口味")
    if any(word in name for word in staple_words):
        tags.append("主食")

    return {
        "calories": calories,
        "protein": protein,
        "fat": fat,
        "price": price,
        "spicy": spicy,
        "type": dish_type,
        "taste": taste,
        "tags": ";".join(dict.fromkeys(tags)),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    labels = parse_labels()

    with LABELS_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "name", "english_name"])
        writer.writeheader()
        writer.writerows(labels)

    with DISHES_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["name", "calories", "protein", "fat", "price", "spicy", "type", "taste", "tags"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for label in labels:
            writer.writerow({"name": label["name"], **infer_profile(label["name"])})

    print(f"wrote {LABELS_PATH}")
    print(f"wrote {DISHES_PATH}")


if __name__ == "__main__":
    main()
