# 自定义食堂菜品图片数据集

如果要训练和系统菜品名称完全一致的模型，把图片按类别文件夹放到这里：

```text
food_images/
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
python backend/train_food_classifier.py --source local --epochs 8
```

建议每类至少 20-50 张图片。类别文件夹名最好和 `backend/data/dishes.csv` 里的菜品名一致，这样识别结果可以直接联动推荐。
