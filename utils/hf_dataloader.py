"""
Hugging Face 数据集加载器
支持从本地 hf_datasets 加载数据
"""

import os

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
import cv2
from utils.utils import cvtColor, preprocess_input


class HFUnetDataset(Dataset):
    """从 Hugging Face 格式数据集加载的 Unet Dataset"""
    
    # 类别名称到索引的映射
    CLASS_TO_IDX = {
        "动物类": 0,
        "植物类": 1,
        "复合类": 2,
    }
    
    def __init__(
        self,
        data_dir,
        input_shape,
        num_classes,
        augmentation=True,
        split="train",
        config="full",
        task: str = "multiclass",
        cache_dir: str | None = None,
        return_cls_label: bool = False,
    ):
        """
        Args:
            data_dir: hf_datasets 目录路径 (如 "./hf_datasets/merged_dataset_v2")
            input_shape: 输入图像尺寸 [H, W]
            num_classes: 类别数
            augmentation: 是否数据增强
            split: 数据分割 ("train", "validation", "test")
            config: 数据集配置 ("full" 或 "no-ai")
            task: 任务类型 ("binary" 或 "multiclass")
            cache_dir: Hugging Face datasets 缓存目录（建议放在项目内，避免无权限写入 $HOME/.cache）
            return_cls_label: 是否返回分类标签（用于多任务学习）
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.task = task
        self.return_cls_label = return_cls_label
        
        # 加载 Hugging Face 数据集
        dataset_path = f"{data_dir}/{config}"
        cache_dir = cache_dir or os.environ.get("HF_DATASETS_CACHE") or ".hf-cache/datasets"
        os.makedirs(cache_dir, exist_ok=True)
        self.dataset = load_dataset(dataset_path, split=split, cache_dir=cache_dir)
        self.length = len(self.dataset)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # 从 Hugging Face 数据集获取样本
        sample = self.dataset[index]
        
        # 获取图像和掩码 (PIL Image)
        jpg = sample["image"].convert("RGB")
        png = sample["mask"].convert("L")
        
        # 数据增强
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.augmentation)
        
        # 图像预处理
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)

        # 二分类：将所有前景类别合并为 1（前景 vs 背景）
        if self.task == "binary":
            png = (png > 0).astype(np.uint8)
        
        # 将标签值大于类别数的部分设置为类别数（忽略这些区域）
        png[png >= self.num_classes] = self.num_classes
        
        # 将标签转换为one-hot编码
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        
        # 如果需要返回分类标签
        if self.return_cls_label:
            label_name = sample.get("label", "unknown")
            # 从 label 字段提取类别（如 "动物类100" -> "动物类"）
            for class_name in self.CLASS_TO_IDX.keys():
                if label_name.startswith(class_name):
                    cls_label = self.CLASS_TO_IDX[class_name]
                    break
            else:
                cls_label = 0  # 默认动物类
            return jpg, png, seg_labels, cls_label
        
        return jpg, png, seg_labels
    
    def rand(self, a=0, b=1):
        """生成随机数"""
        return np.random.rand() * (b - a) + a
    
    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        """数据增强"""
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        
        iw, ih = image.size
        h, w = input_shape
        
        if not random:
            # 验证模式：等比例缩放并中心裁剪
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            
            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label
        
        # 训练模式：随机数据增强
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
            
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        
        # 随机翻转
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 随机位置粘贴
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label
        
        # HSV 色彩增强
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        
        hue_channel, sat_channel, val_channel = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        image_data = cv2.merge((cv2.LUT(hue_channel, lut_hue), 
                                cv2.LUT(sat_channel, lut_sat), 
                                cv2.LUT(val_channel, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


def hf_unet_dataset_collate(batch):
    """DataLoader 的 collate_fn"""
    # 判断是否是多任务模式（batch 元素长度为 4）
    is_multitask = len(batch[0]) == 4
    
    images = []
    pngs = []
    seg_labels = []
    cls_labels = [] if is_multitask else None
    
    if is_multitask:
        for img, png, labels, cls_label in batch:
            images.append(img)
            pngs.append(png)
            seg_labels.append(labels)
            cls_labels.append(cls_label)
    else:
        for img, png, labels in batch:
            images.append(img)
            pngs.append(png)
            seg_labels.append(labels)
    
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    
    if is_multitask:
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        return images, pngs, seg_labels, cls_labels
    
    return images, pngs, seg_labels
