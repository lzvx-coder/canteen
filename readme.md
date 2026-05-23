# 智能食堂菜品推荐与营养分析系统

## 1. 项目简介

本项目实现了一个面向校园食堂场景的智能菜品推荐系统。系统根据学生输入的预算、饮食目标、口味偏好、忌口和最近饮食记录，使用多目标加权评分算法生成 Top3 推荐套餐，并输出热量、蛋白质、脂肪和价格等营养分析结果。

本版本加入了可训练的 PyTorch 菜品图像分类模块。训练脚本使用 MobileNetV3-Small 迁移学习，可以：

- 从网上自动下载 Food-101 数据集训练通用食物分类模型。
- 使用本地 `backend/data/food_images/菜名/*.jpg` 训练和食堂菜名一致的自定义模型。
- 将训练后的 `backend/models/food_classifier.pt` 自动接入 `/recognize` 接口。
- 没有模型文件时，系统仍会使用轻量规则识别兜底，前端不会报错。

## 2. 实现功能

- 菜品数据库：内置 40 个校园食堂常见菜品，包含热量、蛋白质、脂肪、价格、辣度、类型、口味和标签。
- 用户画像输入：支持预算、减脂/增肌/普通/控预算、喜欢口味、忌口、最近吃过菜品。
- 智能推荐算法：综合营养匹配、口味匹配、预算匹配、健康评分和历史重复惩罚进行排序。
- PyTorch 图像识别：上传菜品图片后，后端优先使用训练好的 MobileNetV3 模型预测菜品类别。
- Food-101 数据集支持：可通过 `torchvision.datasets.Food101(download=True)` 从网上自动下载公开数据集。
- 本地中文菜品数据支持：可将图片按菜名文件夹放置，训练出与 `dishes.csv` 直接对应的分类器。
- 识别联动推荐：识别到的菜品会自动写入“最近吃过”，并根据识别菜品口味刷新推荐结果。
- 可视化展示：使用 ECharts 展示 Top3 套餐营养对比和 TOP1 分项评分雷达图。

## 3. 环境安装

进入项目目录：

```bash
cd 
```

安装 PyTorch 相关依赖：

```bash
pip install -r backend/requirements.txt
```

如果你有 NVIDIA 显卡，建议按 PyTorch 官网给出的 CUDA 版本命令安装 `torch` 和 `torchvision`，训练会快很多。

## 4. 训练模型

### 方式 A：使用本地中文菜品图片训练

目录结构如下：

```text
backend/data/food_images/
├── 西兰花鸡胸肉/
│   ├── 001.jpg
│   └── 002.jpg
├── 番茄炒蛋/
│   ├── 001.jpg
│   └── 002.jpg
└── 麻婆豆腐/
    ├── 001.jpg
    └── 002.jpg
```

训练命令：

```bash
python backend/train_food_classifier.py --source local --epochs 8 --batch-size 16
```

建议每类至少 20-50 张图片。类别文件夹名最好和 `backend/data/dishes.csv` 中的菜名一致，这样模型预测结果可以直接联动推荐。

### 方式 B：从网上下载 Food-101 训练

Food-101 是公开食物图片分类数据集，包含 101 类食物。脚本会通过 `torchvision` 自动下载：

```bash
python backend/train_food_classifier.py --source food101 --epochs 5 --batch-size 16 --max-classes 20
```

说明：

- Food-101 全量数据较大，下载和训练时间较长。
- `--max-classes 20` 表示先取前 20 类做课程演示，速度更快。
- 训练完成后会生成 `backend/models/food_classifier.pt`。
- Food-101 类别是英文西餐类别，系统会把部分类别映射到食堂菜品，例如 `chicken_curry -> 咖喱鸡块`，`omelette -> 番茄炒蛋`，`steak -> 黑椒牛柳`。

### 常用训练参数

```bash
python backend/train_food_classifier.py --help
```

常用参数：

- `--source local`：使用本地中文菜品文件夹训练。
- `--source food101`：自动下载并训练 Food-101。
- `--epochs 8`：训练轮数。
- `--batch-size 16`：批大小，显存不足可改小。
- `--max-classes 20`：Food-101 演示时限制类别数。
- `--max-per-class 50`：本地数据很多时限制每类图片数量。
- `--no-pretrained`：不使用 ImageNet 预训练权重。

## 5. 运行系统

### 本机演示

```bash
python app.py
```

系统会自动打开浏览器。本机演示地址会带 `auto_shutdown=1`，关闭页面时会自动关闭后端服务。

## 6. 推荐算法

系统采用多目标加权评分：

```text
score =
0.35 * 营养匹配
+ 0.25 * 口味匹配
+ 0.18 * 预算匹配
+ 0.14 * 健康评分
+ 0.08 * 历史避免重复
```

其中：

- 营养匹配：根据目标判断热量、蛋白质、脂肪是否落在合理区间。
- 口味匹配：判断套餐是否符合喜欢口味，并对忌口内容进行扣分。
- 预算匹配：套餐价格越接近且不超过预算，得分越高。
- 健康评分：鼓励低脂、高蛋白、包含蔬菜或汤品的组合。
- 历史避免重复：最近吃过的菜品会降低权重，体现个性化推荐。

## 7. 图像识别流程

```text
上传菜品图片
-> PyTorch MobileNetV3 分类器预测 TopK 类别
-> 将预测类别映射到食堂菜品
-> 如果没有训练模型，则使用前端像素特征 + 后端规则识别兜底
-> 将最佳识别菜品写入最近吃过
-> 触发个性化推荐与营养分析
```

## 8. 项目结构

```text
exp4/
├── app.py
├── backend/
│   ├── app.py
│   ├── ml_inference.py
│   ├── train_food_classifier.py
│   ├── requirements.txt
│   ├── models/
│   │   └── food_classifier.pt
│   └── data/
│       ├── dishes.csv
│       └── food_images/
├── frontend/
│   └── index.html
└── readme.md
```

## 9. 可写入报告的创新点

- 将食堂点餐问题转化为多目标优化问题。
- 使用用户画像实现个性化推荐。
- 引入历史饮食避免重复机制，提升推荐多样性。
- 使用 PyTorch 迁移学习训练菜品图像分类模型，实现真实机器学习流程。
- 支持 Food-101 公开数据集下载训练，也支持本地中文菜品数据集训练。
- 将“图像感知 -> 菜品识别 -> 营养分析 -> 个性化推荐”串成完整智能应用闭环。
