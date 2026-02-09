import os

import numpy as np
import torch

from model.unet_resnet import Unet as UNetResNet50
from model.unet_plain import UNetPlain
from model.unet_attention import AttentionUNet
from model.unet_dualdense import DualDenseUNet
from model.unet_multitask import MultiTaskUNet


SUPPORTED_MODELS = {
    "unet_plain": UNetPlain,
    "unet_resnet50": UNetResNet50,
    "attention_unet": AttentionUNet,
    "dualdense_unet": DualDenseUNet,
    "multitask_unet": MultiTaskUNet,
}


def build_model(model_name: str, num_classes: int, num_seg_classes: int = 1, num_cls_classes: int = 3):
    """
    构建模型
    
    Args:
        model_name: 模型名称
        num_classes: 分割类别数 (用于普通模型)
        num_seg_classes: 多任务模型的分割类别数
        num_cls_classes: 多任务模型的分类类别数
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {sorted(SUPPORTED_MODELS.keys())}")
    
    if model_name == "multitask_unet":
        return SUPPORTED_MODELS[model_name](num_seg_classes=num_seg_classes, num_cls_classes=num_cls_classes)
    else:
        return SUPPORTED_MODELS[model_name](num_classes=num_classes)


def load_weights_flexible(model, weights_path: str):
    """
    按 key/shape 匹配加载权重，跳过不匹配的层（常用于 num_classes 不一致的 finetune）。
    """
    if not weights_path:
        return model
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model_dict = model.state_dict()
    pretrained_dict = torch.load(weights_path, map_location="cpu")
    load_key, no_load_key, temp_dict = [], [], {}

    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)

    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded weights: {len(load_key)} keys, Skipped: {len(no_load_key)} keys")
    return model

