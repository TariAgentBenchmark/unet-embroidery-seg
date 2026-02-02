# 导入标准库和第三方库
import os
import torch
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
        task=args.task,
        cache_dir=args.cache_dir,
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

    model = build_model(args.model, num_classes=num_classes)
    
    # 加载权重
    weights_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights_dict)
    model.to(device)

    print(f"Model loaded from: {args.weights}")
    print("Starting evaluation...\n")
    
    if args.task == "binary":
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
    parser.add_argument("--data-config", default="no-ai", choices=["full", "no-ai"],
                        help="Dataset config to use: 'full' or 'no-ai'")
    parser.add_argument("--weights", default="weights/unet_resnet_voc.pth",
                        help="Path to model weights")
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"],
                        help="Segmentation task: 'binary' or 'multiclass'")
    parser.add_argument(
        "--model",
        default="unet_resnet50",
        choices=sorted(SUPPORTED_MODELS.keys()),
        help="Model architecture",
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
