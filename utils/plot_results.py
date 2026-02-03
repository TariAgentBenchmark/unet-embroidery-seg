import os
import logging

# 避免在受限环境下写入 $HOME/.matplotlib 失败
os.environ.setdefault("MPLCONFIGDIR", ".mpl-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 避免 Times New Roman 缺失导致大量 findfont 警告；使用通用 serif fallback
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

def plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder):
    # 准备数据
    epochs = range(1, len(train_losses) + 1)

    def _get_series(key):
        return [float(m.get(key, 0.0)) for m in val_metrics_history]

    # 兼容二分类/多分类两种指标集合
    metric_keys_priority = [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "Accuracy",
        "Pixel Accuracy",
        "Mean Accuracy",
        "Mean IoU",
        "Frequency Weighted IoU",
    ]
    metric_keys = [k for k in metric_keys_priority if len(val_metrics_history) > 0 and k in val_metrics_history[0]]

    # ========================
    # 📈 绘制 Loss 曲线
    # ========================
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(weights_folder, "loss_curve.png"), dpi=300)
    plt.close()

    # =========================
    # 📈 绘制指标曲线
    # =========================
    plt.figure(figsize=(8,6))
    for k in metric_keys:
        plt.plot(epochs, _get_series(k), label=k, linewidth=2)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(weights_folder, "metrics_curve.png"), dpi=300)
    plt.close()
