# 导入标准库和第三方库
import json
import os
from functools import partial
import csv

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import datetime
import subprocess

# 导入自定义模块和模型
from model.model_factory import build_model, load_weights_flexible, SUPPORTED_MODELS
from model.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.hf_dataloader import HFUnetDataset, hf_unet_dataset_collate
from utils.utils import seed_everything, worker_init_fn
from utils.train_and_eval import (
    evaluate,
    evaluate_binary,
    train_one_epoch,
    train_one_epoch_binary,
)

from utils.create_exp_folder import create_exp_folder
from utils.plot_results import plot_training_curves
from utils.vis_export import export_binary_visuals


def get_gpu_usage():
    """GPU占用计算函数"""
    if not torch.cuda.is_available():
        return 0
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        used, total = map(int, result.strip().split(','))
        return used
    except:
        return 0


def create_model(model_name, num_classes, weights):
    """创建模型"""
    model = build_model(model_name, num_classes=num_classes)
    weights_init(model)

    if weights:
        load_weights_flexible(model, weights)
    
    return model


def get_optimizer_and_lr(model, batch_size, train_epoch, momentum, weight_decay):
    """获取优化器和学习率调度器"""
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    lr_decay_type = 'cos'
    nbs = 16
    lr_limit_max = 1e-4
    lr_limit_min = 1e-4

    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                           weight_decay=weight_decay)
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, train_epoch)
    
    return optimizer, lr_scheduler_func


def train(args):
    seed_everything(args.seed)

    if args.task == "binary":
        # 二分类：背景(0) + 前景(1)
        num_classes = 2
    else:
        # 多分类：args.num_classes 为前景类数量（不含背景）
        num_classes = args.num_classes + 1

    train_epoch = args.epochs
    batch_size = args.batch_size
    num_workers = args.workers

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    exp_folder, weights_folder = create_exp_folder()

    os.makedirs(args.cache_dir, exist_ok=True)
    input_shape = [args.input_size, args.input_size]

    # 保存本次实验配置（便于复现与汇总）
    with open(os.path.join(exp_folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 创建 Hugging Face 数据集
    print(f"Loading HF Dataset from: {args.data_path}, config: {args.data_config}")
    train_dataset = HFUnetDataset(
        args.data_path, 
        input_shape, 
        num_classes, 
        augmentation=True, 
        split="train",
        config=args.data_config,
        task=args.task,
        cache_dir=args.cache_dir,
    )
    val_dataset = HFUnetDataset(
        args.data_path, 
        input_shape, 
        num_classes, 
        augmentation=False, 
        split="validation",
        config=args.data_config,
        task=args.task,
        cache_dir=args.cache_dir,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=hf_unet_dataset_collate,
        sampler=None,
        worker_init_fn=partial(worker_init_fn, seed=args.seed)
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=hf_unet_dataset_collate,
        sampler=None,
        worker_init_fn=partial(worker_init_fn, seed=args.seed)
    )

    model = create_model(args.model, num_classes=num_classes, weights=args.weights)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    optimizer, lr_scheduler_func = get_optimizer_and_lr(
        model, batch_size, train_epoch, args.momentum, args.weight_decay
    )

    start_time = time.time()

    best_score = -1.0
    best_epoch = None
    best_val_metrics = None
    best_model_path = os.path.join(weights_folder, "best.pth")
    last_model_path = os.path.join(weights_folder, "last.pth")

    train_losses = []
    val_losses = []
    val_metrics_history = []

    # 二分类 BCE 可选 pos_weight（处理前景/背景不平衡）
    pos_weight = None
    if args.task == "binary" and args.loss == "bce" and args.pos_weight:
        if args.pos_weight == "auto":
            # 基于训练集采样估计 pos_weight = neg/pos
            total_pos = 0
            total_neg = 0
            sample_n = min(args.pos_weight_samples, len(train_dataset))
            idxs = np.linspace(0, len(train_dataset) - 1, sample_n, dtype=int)
            for i in idxs:
                _, png, _ = train_dataset[i]
                total_pos += int((png == 1).sum())
                total_neg += int((png == 0).sum())
            if total_pos > 0:
                pos_weight = torch.tensor([total_neg / total_pos], dtype=torch.float32, device=device)
                print(f"[pos_weight auto] neg/pos = {float(pos_weight.item()):.4f} (samples={sample_n})")
        else:
                pos_weight = torch.tensor([float(args.pos_weight)], dtype=torch.float32, device=device)

    max_train_batches = args.max_train_batches if args.max_train_batches and args.max_train_batches > 0 else None
    max_val_batches = args.max_val_batches if args.max_val_batches and args.max_val_batches > 0 else None
    max_test_batches = args.max_test_batches if args.max_test_batches and args.max_test_batches > 0 else None

    for epoch in range(train_epoch):
        gpu_used = get_gpu_usage()
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        if args.task == "binary":
            loss = train_one_epoch_binary(
                model,
                optimizer,
                train_loader,
                device,
                loss_name=args.loss,
                pos_weight=pos_weight,
                gpu_used=gpu_used,
                scaler=scaler,
                epoch=epoch,
                train_epoch=train_epoch,
                ignore_index=None,
                max_batches=max_train_batches,
            )
        else:
            # 兼容原多分类训练流程（CE/Focal + Dice）
            focal_loss = args.loss == "focal"
            dice_loss = args.use_dice
            loss = train_one_epoch(
                model, optimizer, train_loader, device,
                dice_loss, focal_loss, gpu_used, num_classes, scaler,
                epoch, train_epoch
            )

        train_losses.append(loss)

        if args.task == "binary":
            metrics = evaluate_binary(
                model,
                val_loader,
                device,
                loss_name=args.loss,
                pos_weight=pos_weight,
                ignore_index=None,
                max_batches=max_val_batches,
            )
            current_score = float(metrics["IoU"])
        else:
            metrics = evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes)
            current_score = float(metrics["Mean IoU"])

        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            best_val_metrics = metrics
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with score: {best_score:.4f}")

        torch.save(model.state_dict(), last_model_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training completed in {total_time_str}")

    plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder)

    # 评估 best checkpoint 在 test 集上的效果（用于生成对比表格）
    test_metrics = None
    try:
        test_dataset = HFUnetDataset(
            args.data_path,
            input_shape,
            num_classes,
            augmentation=False,
            split="test",
            config=args.data_config,
            task=args.task,
            cache_dir=args.cache_dir,
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=max(0, num_workers // 2),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=hf_unet_dataset_collate,
            sampler=None,
        )
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        if args.task == "binary":
            test_metrics = evaluate_binary(
                model,
                test_loader,
                device,
                loss_name=args.loss,
                pos_weight=pos_weight,
                ignore_index=None,
                max_batches=max_test_batches,
            )
        else:
            test_metrics = evaluate(model, test_loader, device, dice_loss=True, focal_loss=False, num_classes=num_classes)
        with open(os.path.join(exp_folder, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        # 导出固定样例的可视化结果（用于论文对比图）
        if args.task == "binary" and args.export_vis:
            export_binary_visuals(
                model=model,
                hf_unet_dataset=test_dataset,
                out_dir=os.path.join(exp_folder, "vis"),
                input_shape=input_shape,
                device=device,
                num_samples=args.vis_num,
                seed=args.vis_seed,
            )
    except Exception as e:
        print(f"[WARN] Skip test evaluation: {e}")

    with open(os.path.join(exp_folder, "val_metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics_history, f, ensure_ascii=False, indent=2)

    # 同步保存一份 CSV，方便直接粘贴到论文表格/Excel
    csv_path = os.path.join(exp_folder, "val_metrics_history.csv")
    fieldnames = ["epoch"]
    for m in val_metrics_history:
        for k in m.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, m in enumerate(val_metrics_history, start=1):
            row = {"epoch": i}
            row.update(m)
            writer.writerow(row)
    with open(os.path.join(exp_folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_score": float(best_score),
                "best_val_metrics": best_val_metrics,
                "test_metrics": test_metrics,
                "best_model_path": best_model_path,
                "last_model_path": last_model_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="U-Net Training with HF Dataset")
    
    parser.add_argument("--weights", default="weights/unet_resnet_voc.pth",
                        help="Path to the pretrained model weights")
    parser.add_argument("--data-path", default="./hf_datasets/merged_dataset_v2", 
                        help="Path to HF dataset directory")
    parser.add_argument("--data-config", default="no-ai", choices=["full", "no-ai"],
                        help="Dataset config to use: 'full' or 'no-ai'")
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"],
                        help="Segmentation task: 'binary' (foreground/background) or 'multiclass'")
    parser.add_argument(
        "--model",
        default="unet_resnet50",
        choices=sorted(SUPPORTED_MODELS.keys()),
        help="Model architecture",
    )
    parser.add_argument("--loss", default="lovasz_hinge",
                        choices=["bce", "lovasz_hinge", "ce", "focal"],
                        help="Loss function. For binary: bce/lovasz_hinge. For multiclass: ce/focal (+ optional dice).")
    parser.add_argument("--pos-weight", default="auto",
                        help="For binary BCE only: 'auto' or a float value. Disable by passing empty string.")
    parser.add_argument("--pos-weight-samples", default=80, type=int,
                        help="How many train samples to estimate pos_weight when pos-weight=auto")
    parser.add_argument(
        "--use-dice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For multiclass only: add Dice loss",
    )
    parser.add_argument("--num-classes", default=4, type=int,
                        help="For multiclass only: number of foreground classes (excluding background)")
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="Number of total epochs to train")
    parser.add_argument("--input-size", default=512, type=int,
                        help="Input image size (square)")
    parser.add_argument("--workers", default=4, type=int, metavar="N",
                        help="Number of data loading workers")
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='Weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.cuda.amp for mixed precision training",
    )
    parser.add_argument("--seed", default=11, type=int, help="Random seed")
    parser.add_argument("--cache-dir", default=".hf-cache/datasets",
                        help="Hugging Face datasets cache dir")
    parser.add_argument(
        "--export-vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export visualization grids on test split",
    )
    parser.add_argument("--vis-num", default=8, type=int, help="How many samples to export")
    parser.add_argument("--vis-seed", default=0, type=int, help="Random seed for visualization sampling")
    parser.add_argument("--max-train-batches", default=0, type=int, help="Limit train batches per epoch (0=all)")
    parser.add_argument("--max-val-batches", default=0, type=int, help="Limit val batches per epoch (0=all)")
    parser.add_argument("--max-test-batches", default=0, type=int, help="Limit test batches (0=all)")
    
    args = parser.parse_args()
    if args.pos_weight == "":
        args.pos_weight = None
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
