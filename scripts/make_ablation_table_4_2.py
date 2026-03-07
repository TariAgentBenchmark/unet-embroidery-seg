import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = ["Dice", "IoU", "Precision", "Recall", "Accuracy"]
ABLATION_COMBINATIONS = [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(runs_dir: Path):
    runs = []
    for exp_dir in sorted(runs_dir.glob("exp*")):
        config_path = exp_dir / "config.json"
        summary_path = exp_dir / "summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue

        config = read_json(config_path)
        summary = read_json(summary_path)
        test_metrics = summary.get("test_metrics")
        if test_metrics is None:
            test_path = exp_dir / "test_metrics.json"
            if test_path.exists():
                test_metrics = read_json(test_path)

        runs.append(
            {
                "exp_name": exp_dir.name,
                "config": config,
                "summary": summary,
                "test_metrics": test_metrics or {},
            }
        )
    return runs


def best_by_iou(runs):
    best_run = None
    best_iou = None
    for run in runs:
        iou = run["test_metrics"].get("IoU")
        if iou is None:
            continue
        iou = float(iou)
        if best_run is None or iou > best_iou:
            best_run = run
            best_iou = iou
    return best_run


def format_percent(value):
    if value is None:
        return ""
    return f"{float(value) * 100.0:.2f}"


def matches(run, args):
    config = run["config"]
    return (
        config.get("task") == args.task
        and config.get("model") == args.model
        and config.get("loss") == args.loss
        and config.get("data_config") == args.data_config
    )


def build_row(run, use_aspp: bool, use_eca: bool, use_sa: bool):
    row = {
        "Model": "U-Net",
        "ASPP": "yes" if use_aspp else "no",
        "ECA": "yes" if use_eca else "no",
        "SA": "yes" if use_sa else "no",
        "exp": run["exp_name"] if run else "",
    }
    metrics = run["test_metrics"] if run else {}
    for metric_key in METRIC_KEYS:
        row[metric_key] = format_percent(metrics.get(metric_key))
    return row


def main():
    parser = argparse.ArgumentParser(description="Generate Table 4-2 module ablation CSV from run/train experiments.")
    parser.add_argument("--runs-dir", default="run/train", help="Directory containing exp*/ folders")
    parser.add_argument("--output", default="run/tables/table_4_2_ablation.csv", help="Output CSV path")
    parser.add_argument("--data-config", default="no-ai", choices=["no-ai", "full", "sam3", "sam3-label"])
    parser.add_argument("--task", default="binary", choices=["binary"])
    parser.add_argument("--model", default="unet_plain", help="Model used for module ablation")
    parser.add_argument("--loss", default="lovasz_hinge", help="Loss used for module ablation")
    args = parser.parse_args()

    runs = [run for run in collect_runs(Path(args.runs_dir)) if matches(run, args)]

    rows = []
    for use_aspp, use_eca, use_sa in ABLATION_COMBINATIONS:
        matched_runs = [
            run
            for run in runs
            if bool(run["config"].get("use_aspp", False)) == use_aspp
            and bool(run["config"].get("use_eca", False)) == use_eca
            and bool(run["config"].get("use_sa", False)) == use_sa
        ]
        rows.append(build_row(best_by_iou(matched_runs), use_aspp, use_eca, use_sa))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "ASPP", "ECA", "SA", "exp", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
