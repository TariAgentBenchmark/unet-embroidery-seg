# 导入标准库和第三方库
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# 导入自定义模块和模型
from model.model_factory import build_model, SUPPORTED_MODELS
from utils.hf_dataloader import HFUnetDataset, hf_unet_dataset_collate
from utils.train_and_eval import evaluate, evaluate_binary


class LogColor:
    """终端颜色常量"""
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"


def val(args):
    """验证函数"""
    if args.task == "binary":
        num_classes = 2
    elif args.task == "multitask":
        num_classes = 2
    else:
        num_classes = args.num_classes + 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.cache_dir, exist_ok=True)
    input_shape = [args.input_size, args.input_size]

    # 加载 Hugging Face 数据集
    print(f"Loading HF Dataset from: {args.data_path}, config: {args.data_config}, split: test")
    val_dataset = HFUnetDataset(
        args.data_path,
        input_shape,
        num_classes,
        augmentation=False,
        split="test",
        config=args.data_config,
        task="binary" if args.task == "multitask" else args.task,
        cache_dir=args.cache_dir,
        return_cls_label=(args.task == "multitask"),
    )
    
    print(f"Test samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=hf_unet_dataset_collate,
        sampler=None
    )

    # 创建模型
    if args.task == "multitask":
        model = build_model(args.model, num_classes=1, num_seg_classes=1, num_cls_classes=3)
    else:
        model = build_model(args.model, num_classes=num_classes)
    
    # 加载权重
    weights_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights_dict)
    model.to(device)

    print(f"Model loaded from: {args.weights}")
    print("Starting evaluation...\n")
    
    if args.task == "multitask":
        # 多任务验证
        model.eval()
        correct = 0
        total = 0
        seg_preds_list = []
        seg_targets_list = []
        cls_preds_list = []
        cls_targets_list = []
        
        class_names = ["动物类", "植物类", "复合类"]
        
        with torch.no_grad():
            for batch in val_loader:
                images, seg_targets, _, cls_targets = batch
                images = images.to(device)
                seg_targets = seg_targets.to(device)
                
                seg_logits, cls_logits = model(images)
                
                # 分类预测
                _, predicted = cls_logits.max(1)
                total += cls_targets.size(0)
                correct += predicted.eq(cls_targets.to(device)).sum().item()
                cls_preds_list.extend(predicted.cpu().numpy())
                cls_targets_list.extend(cls_targets.numpy())
                
                # 分割预测
                seg_preds = (torch.sigmoid(seg_logits) > 0.5).squeeze(1).cpu().numpy()
                seg_preds_list.extend(seg_preds)
                seg_targets_list.extend(seg_targets.cpu().numpy())
        
        # 计算分割指标
        seg_preds = np.array(seg_preds_list)
        seg_targets = np.array(seg_targets_list)
        intersection = ((seg_preds == 1) & (seg_targets == 1)).sum()
        union = ((seg_preds == 1) | (seg_targets == 1)).sum()
        iou = intersection / (union + 1e-6)
        dice = 2 * intersection / (seg_preds.sum() + seg_targets.sum() + 1e-6)
        cls_acc = 100. * correct / total
        
        # 计算每个类别的准确率
        cls_preds = np.array(cls_preds_list)
        cls_targets = np.array(cls_targets_list)
        
        print("=" * 50)
        print(f"{LogColor.BLUE}Multi-Task Evaluation Results{LogColor.RESET}")
        print("=" * 50)
        print(f"\n{LogColor.RED}Segmentation Metrics:{LogColor.RESET}")
        print(f"  IoU:  {iou:.4f}")
        print(f"  Dice: {dice:.4f}")
        
        print(f"\n{LogColor.RED}Classification Metrics:{LogColor.RESET}")
        print(f"  Overall Accuracy: {cls_acc:.2f}%")
        print(f"\n  Per-Class Accuracy:")
        for i, name in enumerate(class_names):
            mask = cls_targets == i
            if mask.sum() > 0:
                acc = (cls_preds[mask] == i).sum() / mask.sum() * 100
                print(f"    {name}: {acc:.2f}% ({mask.sum()} samples)")
        print("=" * 50)
        
    elif args.task == "binary":
        metrics = evaluate_binary(model, val_loader, device, loss_name=args.loss, pos_weight=None, ignore_index=None)
        print(
            f"{LogColor.RED}Dice{LogColor.RESET}\t"
            f"{LogColor.RED}IoU{LogColor.RESET}\t"
            f"{LogColor.RED}Precision{LogColor.RESET}\t"
            f"{LogColor.RED}Recall{LogColor.RESET}\t"
            f"{LogColor.RED}Accuracy{LogColor.RESET}"
        )
        print(
            f"{metrics['Dice']:.4f}\t"
            f"{metrics['IoU']:.4f}\t"
            f"{metrics['Precision']:.4f}\t"
            f"{metrics['Recall']:.4f}\t"
            f"{metrics['Accuracy']:.4f}"
        )
    else:
        metrics = evaluate(model, val_loader, device, dice_loss=True, focal_loss=False, num_classes=num_classes)
        print(metrics)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="U-Net Validation with HF Dataset")

    parser.add_argument("--data-path", default="./hf_datasets/merged_dataset_v2",
                        help="Path to HF dataset directory")
    parser.add_argument("--data-config", default="no-ai", choices=["full", "no-ai", "sam3"],
                        help="Dataset config to use: 'full', 'no-ai', or 'sam3'")
    parser.add_argument("--weights", default="weights/unet_resnet_voc.pth",
                        help="Path to model weights")
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass", "multitask"],
                        help="Segmentation task: 'binary', 'multiclass', or 'multitask'")
    parser.add_argument(
        "--model",
        default="unet_resnet50",
        choices=sorted(SUPPORTED_MODELS.keys()),
        help="Model architecture (use 'multitask_unet' for multitask)",
    )
    parser.add_argument("--loss", default="lovasz_hinge", choices=["bce", "lovasz_hinge", "ce", "focal"],
                        help="Loss name (only used to report Loss for binary)")
    parser.add_argument("--num-classes", default=4, type=int,
                        help="For multiclass: number of foreground classes (excluding background)")
    parser.add_argument("--input-size", default=512, type=int,
                        help="Input image size (square)")
    parser.add_argument("--cache-dir", default=".hf-cache/datasets",
                        help="Hugging Face datasets cache dir")
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    val(args)
