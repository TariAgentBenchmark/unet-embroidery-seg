# U-Net 刺绣图像分割

基于 U-Net 的刺绣图案语义分割项目，支持从 Hugging Face 数据集加载数据。

## 项目结构

```
.
├── train.py              # 训练脚本
├── val.py                # 验证脚本
├── predict.py            # 预测脚本
├── model/                # U-Net 模型定义
├── utils/                # 工具函数
├── raw_datasets/         # 原始 VOC 数据集
├── hf_datasets/          # HF 格式数据集
└── weights/              # 模型权重
```

## 环境准备

```bash
# 使用 uv 安装依赖
uv sync
```

## 数据集

项目支持两种数据集配置：
- `full`: 完整数据集 (835张)
- `no-ai`: 去除AI生成图像 (763张)

数据已转换为 Hugging Face 格式，存放在 `hf_datasets/merged_dataset_v2/`。

## 运行命令

### 一键跑完整流程（loss 对比 + 4 模型对比 + 消融 + 表格）

```bash
bash run.sh --device cuda --data-config no-ai --epochs 50 --batch-size 16 --input-size 512
```

### 训练

```bash
# 二分类（前景 vs 背景，默认 no-ai）
python train.py --task binary --data-config no-ai --model unet_resnet50 --loss lovasz_hinge --epochs 50 --batch-size 8

# 使用 full 数据集训练
python train.py --data-config full --epochs 50 --batch-size 8

# 使用 no-ai 数据集训练
python train.py --data-config no-ai --epochs 50 --batch-size 8

# 从预训练权重继续训练
python train.py --weights weights/unet_resnet_voc.pth --data-config full
```

### 验证

```bash
python val.py --task binary --data-config no-ai --model unet_resnet50 --weights run/train/exp1/weights/best.pth
```

### 预测

```bash
python predict.py --model unet_resnet50 --weights run/train/exp1/weights/best.pth --data_path input.jpg --num-classes 1
```

### 生成论文表格（CSV）

训练完成后，运行：

```bash
python scripts/make_tables.py --data-config no-ai
```

输出在 `run/tables/`：
- `table_3_1_loss_compare.csv`
- `table_3_2_model_compare.csv`
- `table_4_2_ablation.csv`

## 上传到 Hugging Face

如需重新转换上传数据集：

```bash
export HF_TOKEN="your_token"
uv run python convert_and_upload.py
```

数据集地址：https://huggingface.co/datasets/tari-tech/13803867589-unet-image-seg

## 标签类别

- 植物类
- 动物类
- 生活类
- 复合类
