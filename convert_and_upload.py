#!/usr/bin/env python3
"""
VOC 数据集转换为 Hugging Face 格式并上传

最终整合脚本 - 包含所有功能:
1. VOC 数据集转换为 Hugging Face 格式
2. 创建多 Config 结构 (full + no-ai)
3. 上传到 Hugging Face Hub

使用方法:
    uv run python convert_and_upload.py

需要设置环境变量:
    export HF_TOKEN="your_huggingface_token"
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List
from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import login, HfApi, create_repo


# ============ 配置 ============
VOC_ORIGINAL = Path("./raw_datasets/VOCdevkit/VOC2012")
VOC_NO_AI = Path("./raw_datasets/VOCdevkit_no_ai/VOC2012")
VOC_SAM3 = Path("./raw_datasets/VOCdevkit_SAM3/VOC2012")  # SAM3 标注的 mask
OUTPUT_DIR = Path("./hf_datasets")
REPO_ID = "tari-tech/13803867589-unet-image-seg"


# ============ 转换功能 ============

def get_label_from_filename(filename: str) -> str:
    """从文件名中提取类别标签"""
    name = Path(filename).stem
    label = ""
    for char in name:
        if char.isdigit():
            break
        label += char
    return label if label else "unknown"


def load_split_ids(split_file: Path) -> List[str]:
    """加载分割文件的ID列表"""
    with open(split_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def find_image_path(jpeg_dir: Path, image_id: str) -> Path:
    """在 JPEGImages 目录中查找图片文件"""
    for ext in ['.png', '.jpg', '.jpeg']:
        path = jpeg_dir / f"{image_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"找不到图片: {image_id}")


def find_mask_path(mask_dir: Path, image_id: str) -> Path:
    """在 SegmentationClass 目录中查找掩码文件"""
    for ext in ['.png', '.jpg', '.jpeg']:
        path = mask_dir / f"{image_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"找不到掩码: {image_id}")


def create_dataset_split(
    split_name: str,
    split_ids: List[str],
    jpeg_dir: Path,
    mask_dir: Path,
    subset_name: str
) -> Dataset:
    """创建一个数据分割的 Dataset"""
    print(f"  处理 {split_name} 集, 共 {len(split_ids)} 个样本...")
    
    data = {
        'image': [],
        'mask': [],
        'label': [],
        'filename': [],
        'subset': [],
    }
    
    for image_id in split_ids:
        try:
            img_path = find_image_path(jpeg_dir, image_id)
            mask_path = find_mask_path(mask_dir, image_id)
            
            data['image'].append(str(img_path))
            data['mask'].append(str(mask_path))
            data['label'].append(get_label_from_filename(image_id))
            data['filename'].append(image_id)
            data['subset'].append(subset_name)
        except FileNotFoundError:
            pass
    
    from datasets import Image
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column('image', Image())
    dataset = dataset.cast_column('mask', Image())
    
    return dataset


def convert_voc_to_hf(voc_root: Path, subset_name: str) -> DatasetDict:
    """将 VOC 数据集转换为 Hugging Face 格式"""
    print(f"\n转换 {subset_name} 数据集...")
    
    jpeg_dir = voc_root / 'JPEGImages'
    mask_dir = voc_root / 'SegmentationClass'
    split_dir = voc_root / 'ImageSets' / 'Segmentation'
    
    dataset_dict = DatasetDict()
    
    splits = {
        'train': split_dir / 'train.txt',
        'validation': split_dir / 'val.txt',
        'test': split_dir / 'test.txt',
    }
    
    for split_name, split_file in splits.items():
        if split_file.exists():
            split_ids = load_split_ids(split_file)
            dataset = create_dataset_split(
                split_name, split_ids, jpeg_dir, mask_dir, subset_name
            )
            dataset_dict[split_name] = dataset
            print(f"    {split_name}: {len(dataset)} 个样本")
    
    return dataset_dict


def convert_sam3_to_hf(sam3_root: Path, no_ai_root: Path, subset_name: str) -> DatasetDict:
    """将 SAM3 标注的数据集转换为 Hugging Face 格式
    
    SAM3 数据集特点：
    - mask 保存在 sam3_root/JPEGImages/
    - 原图使用 no_ai_root/JPEGImages/ 中对应的图片
    - 分割信息使用 no_ai_root 的 ImageSets，但只保留有 SAM3 mask 的图片
    """
    print(f"\n转换 {subset_name} 数据集...")
    
    sam3_mask_dir = sam3_root / 'JPEGImages'
    jpeg_dir = no_ai_root / 'JPEGImages'
    split_dir = no_ai_root / 'ImageSets' / 'Segmentation'
    
    # 获取所有 SAM3 标注的 mask 文件名（不含扩展名）
    sam3_image_ids = set()
    for mask_file in sam3_mask_dir.glob('*.png'):
        sam3_image_ids.add(mask_file.stem)
    
    print(f"  找到 {len(sam3_image_ids)} 个 SAM3 标注的 mask")
    
    dataset_dict = DatasetDict()
    
    splits = {
        'train': split_dir / 'train.txt',
        'validation': split_dir / 'val.txt',
        'test': split_dir / 'test.txt',
    }
    
    for split_name, split_file in splits.items():
        if split_file.exists():
            split_ids = load_split_ids(split_file)
            # 只保留有 SAM3 mask 的图片
            filtered_ids = [id for id in split_ids if id in sam3_image_ids]
            skipped = len(split_ids) - len(filtered_ids)
            
            dataset = create_dataset_split(
                split_name, filtered_ids, jpeg_dir, sam3_mask_dir, subset_name
            )
            dataset_dict[split_name] = dataset
            print(f"    {split_name}: {len(dataset)} 个样本 (跳过 {skipped} 个无 SAM3 标注)")
    
    return dataset_dict


# ============ 创建多 Config 结构 ============

def create_merged_dataset():
    """创建合并的多 Config 数据集"""
    print("=" * 60)
    print("创建多 Config Parquet 数据集")
    print("=" * 60)
    
    merged_dir = OUTPUT_DIR / "merged_dataset_v2"
    
    # 清理旧数据
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换 full
    print("\n1. 转换 full 数据集...")
    full_ds = convert_voc_to_hf(VOC_ORIGINAL, "full")
    full_dir = merged_dir / "full"
    full_dir.mkdir(exist_ok=True)
    for split in full_ds.keys():
        split_dir = full_dir / split
        split_dir.mkdir(exist_ok=True)
        full_ds[split].to_parquet(str(split_dir / "data.parquet"))
    print(f"   已保存到: {full_dir}")
    
    # 转换 no-ai
    print("\n2. 转换 no-ai 数据集...")
    noai_ds = convert_voc_to_hf(VOC_NO_AI, "no-ai")
    noai_dir = merged_dir / "no-ai"
    noai_dir.mkdir(exist_ok=True)
    for split in noai_ds.keys():
        split_dir = noai_dir / split
        split_dir.mkdir(exist_ok=True)
        noai_ds[split].to_parquet(str(split_dir / "data.parquet"))
    print(f"   已保存到: {noai_dir}")
    
    # 转换 sam3
    print("\n3. 转换 sam3 数据集...")
    sam3_ds = convert_sam3_to_hf(VOC_SAM3, VOC_NO_AI, "sam3")
    sam3_dir = merged_dir / "sam3"
    sam3_dir.mkdir(exist_ok=True)
    for split in sam3_ds.keys():
        split_dir = sam3_dir / split
        split_dir.mkdir(exist_ok=True)
        sam3_ds[split].to_parquet(str(split_dir / "data.parquet"))
    print(f"   已保存到: {sam3_dir}")
    
    # 创建 README
    readme_content = '''---
tags:
- image-segmentation
- computer-vision
- embroidery
- unet
- semantic-segmentation
---

# 刺绣图像分割数据集

用于 U-Net 语义分割的刺绣图像数据集。

## 使用方法

```python
from datasets import load_dataset

# 加载完整数据集 (835张)
ds = load_dataset("tari-tech/13803867589-unet-image-seg", data_dir="full")

# 加载去除AI图的数据集 (763张)
ds = load_dataset("tari-tech/13803867589-unet-image-seg", data_dir="no-ai")

# 加载 SAM3 重新标注的数据集 (563张)
ds = load_dataset("tari-tech/13803867589-unet-image-seg", data_dir="sam3")
```

## 数据集统计

| Config | Train | Val | Test | Total |
|--------|-------|-----|------|-------|
| full   | 584   | 167 | 84   | 835   |
| no-ai  | 534   | 152 | 77   | 763   |
| sam3   | ~394  | ~113| ~56  | 563   |

## 子集说明

- **full**: 完整数据集，包含所有图片
- **no-ai**: 去除 AI 生成的图片，只保留真实刺绣图案
- **sam3**: 使用 SAM3 模型重新标注的 mask，基于 no-ai 的子集（部分图片被标注）
'''
    
    with open(merged_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("\n✅ 数据集创建完成!")
    return merged_dir


# ============ 上传功能 ============

def upload_to_hub():
    """上传数据集到 Hugging Face Hub"""
    print("\n" + "=" * 60)
    print("上传到 Hugging Face Hub")
    print("=" * 60)
    
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("错误: 请设置 HF_TOKEN 环境变量")
        sys.exit(1)
    
    merged_dir = OUTPUT_DIR / "merged_dataset_v2"
    
    # 登录
    print("\n1. 登录 Hugging Face...")
    login(token=token)
    
    api = HfApi()
    
    # 清理旧文件
    print("\n2. 清理旧文件...")
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        for f in files:
            if f not in ['.gitattributes']:
                try:
                    api.delete_file(path_in_repo=f, repo_id=REPO_ID, repo_type="dataset")
                    print(f"   删除: {f}")
                except:
                    pass
    except:
        pass
    
    # 上传 README
    print("\n3. 上传 README...")
    api.upload_file(
        path_or_fileobj=str(merged_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    
    # 上传数据
    print("\n4. 上传数据...")
    for config in ["full", "no-ai", "sam3"]:
        print(f"   上传 {config} config...")
        api.upload_folder(
            folder_path=str(merged_dir / config),
            path_in_repo=config,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
    
    print(f"\n✅ 上传完成!")
    print(f"https://huggingface.co/datasets/{REPO_ID}")


# ============ 主函数 ============

def main():
    print("VOC 数据集转换并上传到 Hugging Face Hub")
    print("=" * 60)
    
    # 步骤1: 创建数据集
    create_merged_dataset()
    
    # 步骤2: 上传
    upload_to_hub()
    
    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
