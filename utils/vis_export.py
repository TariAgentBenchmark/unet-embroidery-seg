import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.utils import preprocess_input


def _mask_to_rgb(mask01: np.ndarray, fg_color=(255, 0, 0)) -> np.ndarray:
    mask01 = (mask01 > 0).astype(np.uint8)
    h, w = mask01.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask01 == 1] = np.array(fg_color, dtype=np.uint8)
    return out


def _make_grid(img_rgb: np.ndarray, gt01: np.ndarray, pred01: np.ndarray, alpha: float = 0.5) -> Image.Image:
    img = img_rgb.astype(np.uint8)
    gt_rgb = _mask_to_rgb(gt01, fg_color=(255, 0, 0))
    pred_rgb = _mask_to_rgb(pred01, fg_color=(0, 255, 0))

    overlay = (img.astype(np.float32) * (1 - alpha) + pred_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

    # 2x2: img / gt / pred / overlay
    h, w = img.shape[:2]
    canvas = Image.new("RGB", (w * 2, h * 2))
    canvas.paste(Image.fromarray(img), (0, 0))
    canvas.paste(Image.fromarray(gt_rgb), (w, 0))
    canvas.paste(Image.fromarray(pred_rgb), (0, h))
    canvas.paste(Image.fromarray(overlay), (w, h))
    return canvas


@torch.no_grad()
def export_binary_visuals(
    model,
    hf_unet_dataset,
    out_dir: str,
    input_shape: list[int],
    device: torch.device,
    num_samples: int = 8,
    seed: int = 0,
):
    """
    导出二分类分割可视化：原图 / GT / Pred / Overlay（2x2 grid）

    Args:
        model: torch.nn.Module
        hf_unet_dataset: utils.hf_dataloader.HFUnetDataset 实例（augmentation=False）
        out_dir: 输出目录
        input_shape: [H, W] 输入尺寸（与训练一致）
        device: 推理设备
        num_samples: 导出样本数
        seed: 固定随机种子，保证不同实验可横向对比
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    length = len(hf_unet_dataset)
    num_samples = min(num_samples, length)
    rng = random.Random(seed)
    indices = rng.sample(range(length), k=num_samples) if num_samples > 0 else []

    with (out_path / "indices.json").open("w", encoding="utf-8") as f:
        json.dump(indices, f, ensure_ascii=False, indent=2)

    model = model.eval().to(device)

    for idx in indices:
        sample = hf_unet_dataset.dataset[idx]
        img_pil = sample["image"].convert("RGB")
        mask_pil = sample["mask"].convert("L")

        # 与验证阶段一致：等比例缩放 + padding 到 input_shape
        img_pil, mask_pil = hf_unet_dataset.get_random_data(img_pil, mask_pil, input_shape, random=False)

        img_np = np.array(img_pil, dtype=np.uint8)
        gt = (np.array(mask_pil) > 0).astype(np.uint8)

        img_tensor = np.transpose(preprocess_input(img_np.astype(np.float32)), (2, 0, 1))[None, ...]
        img_tensor = torch.from_numpy(img_tensor).float().to(device)

        logits = model(img_tensor)
        pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        grid = _make_grid(img_np, gt, pred, alpha=0.5)
        filename = sample.get("filename") or f"sample_{idx}"
        save_name = f"{idx:04d}_{Path(filename).stem}_grid.png"
        grid.save(out_path / save_name)
