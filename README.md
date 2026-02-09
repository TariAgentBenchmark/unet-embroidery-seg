# U-Net 刺绣图像分割

基于 U-Net 的刺绣图案语义分割项目，支持从 Hugging Face 数据集加载数据。

## 项目结构

```
.
├── train.py              # 训练脚本（支持二分类/多分类/多任务）
├── val.py                # 验证脚本
├── predict.py            # 预测脚本
├── run.sh                # 一键训练脚本
├── model/                # U-Net 模型定义
│   ├── unet_multitask.py # 多任务模型（分割+分类）
│   └── ...
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

项目支持三种数据集配置：

| 配置 | 说明 | 训练集 | 验证集 | 测试集 | 总计 |
|------|------|--------|--------|--------|------|
| `full` | 完整数据集 | 584 | 167 | 84 | 835 |
| `no-ai` | 去除AI生成图像 | 534 | 152 | 77 | 763 |
| `sam3` | SAM3重新标注的mask | ~394 | ~113 | ~56 | 563 |

数据已转换为 Hugging Face 格式，存放在 `hf_datasets/merged_dataset_v2/`。

## 运行命令

### 一键跑完整流程

#### 二分类（loss 对比 + 4 模型对比 + 消融 + 表格）

```bash
bash run.sh --task binary --device cuda --data-config no-ai --epochs 50 --batch-size 16 --input-size 512
```

#### 多任务训练

```bash
bash run.sh --task multitask --model multitask_unet --data-config sam3 --epochs 50 --batch-size 8
```

如果本地没有 `hf_datasets/merged_dataset_v2/<no-ai|full|sam3>`，`run.sh` 会默认从 Hugging Face 数据集仓库下载到 `hf_datasets/merged_dataset_v2/`（可用 `--hf-repo` / `--hf-local-dir` 覆盖）。

### 训练

#### 二分类分割（前景 vs 背景）

```bash
# 默认 no-ai 数据集
python train.py --task binary --data-config no-ai --model unet_resnet50 --loss lovasz_hinge --epochs 50 --batch-size 8

# 使用 full 数据集
python train.py --data-config full --epochs 50 --batch-size 8

# 使用 sam3 数据集
python train.py --data-config sam3 --epochs 50 --batch-size 8

# 从预训练权重继续训练
python train.py --weights weights/unet_resnet_voc.pth --data-config full
```

#### 多任务学习（分割 + 分类）

同时预测分割mask和图像类别（动物类/植物类/复合类）：

```bash
# 使用 multitask_unet 模型
python train.py \
    --task multitask \
    --model multitask_unet \
    --data-config sam3 \
    --loss bce \
    --cls-loss-weight 1.0 \
    --epochs 50 \
    --batch-size 8
```

参数说明：
- `--task multitask`: 启用多任务模式
- `--model multitask_unet`: 使用多任务模型
- `--cls-loss-weight`: 分类损失权重（默认 1.0）

### 验证

```bash
# 二分类验证
python val.py --task binary --data-config no-ai --model unet_resnet50 --weights run/train/exp1/weights/best.pth

# 多任务验证
python val.py \
    --task multitask \
    --model multitask_unet \
    --data-config sam3 \
    --weights run/train/exp1/weights/best.pth
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
