import argparse
import csv
import json
from pathlib import Path


DEFAULT_METRIC_KEYS = ["Dice", "IoU", "Precision", "Recall", "Accuracy"]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(runs_dir: Path):
    runs = []
    for exp_dir in sorted(runs_dir.glob("exp*")):
        config_path = exp_dir / "config.json"
        summary_path = exp_dir / "summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        config = _read_json(config_path)
        summary = _read_json(summary_path)
        test_metrics = summary.get("test_metrics")
        if test_metrics is None:
            test_path = exp_dir / "test_metrics.json"
            if test_path.exists():
                test_metrics = _read_json(test_path)

        runs.append(
            {
                "exp_dir": str(exp_dir),
                "exp_name": exp_dir.name,
                "config": config,
                "summary": summary,
                "test_metrics": test_metrics or {},
                "best_val_metrics": summary.get("best_val_metrics") or {},
            }
        )
    return runs


def _best_by_metric(runs, metric_key: str):
    best = None
    best_val = None
    for r in runs:
        v = r.get("test_metrics", {}).get(metric_key)
        if v is None:
            continue
        if best is None or float(v) > float(best_val):
            best = r
            best_val = v
    return best


def _write_table_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Generate paper-style tables from run/train experiments.")
    parser.add_argument("--runs-dir", default="run/train", help="Directory that contains exp*/ folders")
    parser.add_argument("--output-dir", default="run/tables", help="Where to write CSV tables")
    parser.add_argument("--data-config", default="no-ai", choices=["no-ai", "full"], help="Filter by dataset config")
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"], help="Filter by task")
    parser.add_argument("--loss-compare-model", default="unet_resnet50", help="Model used for loss comparison table")
    parser.add_argument("--losses", default="bce,lovasz_hinge", help="Comma-separated loss names for Table 3-1")
    parser.add_argument(
        "--models",
        default="unet_plain,unet_resnet50,attention_unet,dualdense_unet",
        help="Comma-separated model names for Table 3-2",
    )
    parser.add_argument(
        "--model-compare-loss",
        default="",
        help="Loss name for Table 3-2. If empty, auto-pick the best loss from Table 3-1 by test IoU.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    runs = collect_runs(runs_dir)

    # Filter base
    def _match_base(r):
        cfg = r["config"]
        return cfg.get("data_config") == args.data_config and cfg.get("task") == args.task

    base_runs = [r for r in runs if _match_base(r)]

    # Table 3-1: loss compare
    losses = [s.strip() for s in args.losses.split(",") if s.strip()]
    loss_runs = [r for r in base_runs if r["config"].get("model") == args.loss_compare_model and r["config"].get("loss") in losses]

    table31 = []
    for loss_name in losses:
        candidates = [r for r in loss_runs if r["config"].get("loss") == loss_name]
        best = _best_by_metric(candidates, "IoU")
        if best is None:
            continue
        row = {"Loss": loss_name, "exp": best["exp_name"]}
        for k in DEFAULT_METRIC_KEYS:
            row[k] = best["test_metrics"].get(k)
        table31.append(row)

    _write_table_csv(out_dir / "table_3_1_loss_compare.csv", table31, ["Loss", "exp", *DEFAULT_METRIC_KEYS])

    # pick best loss for Table 3-2 if not provided
    model_compare_loss = args.model_compare_loss.strip()
    if not model_compare_loss:
        best_loss_run = _best_by_metric(loss_runs, "IoU")
        model_compare_loss = best_loss_run["config"].get("loss") if best_loss_run else (losses[0] if losses else "")

    # Table 3-2: model compare
    models = [s.strip() for s in args.models.split(",") if s.strip()]
    model_runs = [r for r in base_runs if r["config"].get("loss") == model_compare_loss and r["config"].get("model") in models]

    table32 = []
    for model_name in models:
        candidates = [r for r in model_runs if r["config"].get("model") == model_name]
        best = _best_by_metric(candidates, "IoU")
        if best is None:
            continue
        row = {"Model": model_name, "Loss": model_compare_loss, "exp": best["exp_name"]}
        for k in DEFAULT_METRIC_KEYS:
            row[k] = best["test_metrics"].get(k)
        table32.append(row)

    _write_table_csv(out_dir / "table_3_2_model_compare.csv", table32, ["Model", "Loss", "exp", *DEFAULT_METRIC_KEYS])

    # Table 4-2: ablation (loss x attention)
    # Attention off: unet_plain; Attention on: attention_unet
    ablation_losses = losses if losses else ["bce", "lovasz_hinge"]
    ablation_models = [
        ("unet_plain", "no"),
        ("attention_unet", "yes"),
    ]
    table42 = []
    for loss_name in ablation_losses:
        for model_name, attn_flag in ablation_models:
            candidates = [
                r
                for r in base_runs
                if r["config"].get("loss") == loss_name and r["config"].get("model") == model_name
            ]
            best = _best_by_metric(candidates, "IoU")
            if best is None:
                continue
            row = {"Loss": loss_name, "Attention": attn_flag, "Model": model_name, "exp": best["exp_name"]}
            for k in DEFAULT_METRIC_KEYS:
                row[k] = best["test_metrics"].get(k)
            table42.append(row)

    _write_table_csv(out_dir / "table_4_2_ablation.csv", table42, ["Loss", "Attention", "Model", "exp", *DEFAULT_METRIC_KEYS])

    # all runs dump (optional, for debugging)
    all_rows = []
    for r in base_runs:
        cfg = r["config"]
        sm = r["summary"]
        row = {
            "exp": r["exp_name"],
            "model": cfg.get("model"),
            "loss": cfg.get("loss"),
            "data_config": cfg.get("data_config"),
            "task": cfg.get("task"),
            "best_epoch": sm.get("best_epoch"),
            "best_score": sm.get("best_score"),
        }
        for k in DEFAULT_METRIC_KEYS:
            row[f"test_{k}"] = r["test_metrics"].get(k)
        all_rows.append(row)

    _write_table_csv(
        out_dir / "all_runs.csv",
        all_rows,
        ["exp", "model", "loss", "data_config", "task", "best_epoch", "best_score", *[f"test_{k}" for k in DEFAULT_METRIC_KEYS]],
    )


if __name__ == "__main__":
    main()

