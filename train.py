# 导入标准库和第三方库
import os
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import datetime
import subprocess

# 导入自定义模块和模型
from model.unet_resnet import Unet
from model.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.hf_dataloader import HFUnetDataset, hf_unet_dataset_collate
from utils.utils import seed_everything, worker_init_fn
from utils.train_and_eval import train_one_epoch, evaluate

from utils.create_exp_folder import create_exp_folder
from utils.plot_results import plot_training_curves


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


def create_model(num_classes, weights):
    """创建模型"""
    model = Unet(num_classes=num_classes)
    weights_init(model)

    if weights and os.path.exists(weights):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights, map_location='cpu')

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
    seed_everything(11)

    num_classes = args.num_classes + 1
    train_epoch = args.epochs
    batch_size = args.batch_size
    num_workers = args.workers

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    exp_folder, weights_folder = create_exp_folder()

    input_shape = [512, 512]

    # 创建 Hugging Face 数据集
    print(f"Loading HF Dataset from: {args.data_path}, config: {args.data_config}")
    train_dataset = HFUnetDataset(
        args.data_path, 
        input_shape, 
        num_classes, 
        augmentation=True, 
        split="train",
        config=args.data_config
    )
    val_dataset = HFUnetDataset(
        args.data_path, 
        input_shape, 
        num_classes, 
        augmentation=False, 
        split="validation",
        config=args.data_config
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
        worker_init_fn=partial(worker_init_fn, seed=11)
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
        worker_init_fn=partial(worker_init_fn, seed=11)
    )

    model = create_model(num_classes=num_classes, weights=args.weights)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    optimizer, lr_scheduler_func = get_optimizer_and_lr(
        model, batch_size, train_epoch, args.momentum, args.weight_decay
    )

    start_time = time.time()

    best_acc = 0.0
    best_model_path = os.path.join(weights_folder, f"best_model_{args.num_classes}.pth")
    last_model_path = os.path.join(weights_folder, f"last_model_{args.num_classes}.pth")

    train_losses = []
    val_losses = []
    val_metrics_history = []

    focal_loss = False
    dice_loss = True

    for epoch in range(train_epoch):
        gpu_used = get_gpu_usage()
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        loss = train_one_epoch(
            model, optimizer, train_loader, device, 
            dice_loss, focal_loss, gpu_used, num_classes, scaler, 
            epoch, train_epoch
        )

        train_losses.append(loss)

        metrics = evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes)

        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        current_acc = float(metrics["Mean Accuracy"])

        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with acc: {best_acc:.4f}")

        torch.save(model.state_dict(), last_model_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training completed in {total_time_str}")

    plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="U-Net Training with HF Dataset")
    
    parser.add_argument("--weights", default="weights/unet_resnet_voc.pth",
                        help="Path to the pretrained model weights")
    parser.add_argument("--data-path", default="./hf_datasets/merged_dataset_v2", 
                        help="Path to HF dataset directory")
    parser.add_argument("--data-config", default="full", choices=["full", "no-ai"],
                        help="Dataset config to use: 'full' or 'no-ai'")
    parser.add_argument("--num-classes", default=5, type=int,
                        help="Number of classes (excluding background)")
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="Number of total epochs to train")
    parser.add_argument("--workers", default=4, type=int, metavar="N",
                        help="Number of data loading workers")
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='Weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--amp", default=True, type=bool, 
                        help="Use torch.cuda.amp for mixed precision training")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
