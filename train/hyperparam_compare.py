#!/usr/bin/env python3
"""
YOLOv8 hyperparameter checkpoint comparison for bottle cap detection.

Goal:
- Train every LR x batch config for a fixed number of epochs.
- Save checkpoint metrics at selected epoch checkpoints.
- Generate CSV summaries, a text report, and comparison graphs.

Experiments:
- lr0: 0.001, 0.003, 0.005
- batch: 8 by default, because batch size was already selected
- epochs: 150 by default, because the best range was around 135-150

YOLO augmentation is kept at Ultralytics defaults for this phase so the result
reflects learning rate, batch size, and epoch effects only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================
# CONFIG
# ==============================================================
BASE_DIR = Path(__file__).parent.resolve()

BOTTLE_PROJECT_DIR = Path(
    r"C:\Users\User\Desktop\digital image\Project Bottle Cap Detection\Bottle Cap Detection All"
)
TRAINRUN_DIR = BOTTLE_PROJECT_DIR / "Trainandrun"
DATASET_DIR = TRAINRUN_DIR / "Data label from roboflow"
DATA_YAML = DATASET_DIR / "data.yaml"
YOLO_MODEL = TRAINRUN_DIR / "yolov8n.pt"

PREVIOUS_RUN_DIR = TRAINRUN_DIR / "runs" / "bottle_cap"
PREVIOUS_WEIGHT = PREVIOUS_RUN_DIR / "weights" / "best.pt"

OUTPUT_DIR = BASE_DIR / "yolo_hyperparam_output"

CLASS_NAMES = ["closed_cap", "half_cap", "no_cap"]
LR_CANDIDATES = [0.001, 0.003, 0.005]
BATCH_CANDIDATES = [8]
CHECKPOINTS = [50, 100, 150]
MAX_EPOCHS = 150

IMAGE_SIZE = 640
SEED = 42
WORKERS = 8
LRF = 0.01
OPTIMIZER = "SGD"


# ==============================================================
# DATASET HELPERS
# ==============================================================
def ensure_dataset() -> None:
    required = [
        DATASET_DIR / "train" / "images",
        DATASET_DIR / "train" / "labels",
        DATASET_DIR / "valid" / "images",
        DATASET_DIR / "valid" / "labels",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("[ERROR] Missing YOLO dataset folders:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)


def write_data_yaml() -> None:
    content = (
        "# YOLOv8 dataset config\n"
        f"path: {DATASET_DIR}\n\n"
        "train: train/images\n"
        "val: valid/images\n\n"
        "nc: 3\n"
        "names:\n"
        "  0: closed_cap\n"
        "  1: half_cap\n"
        "  2: no_cap\n"
    )
    DATA_YAML.write_text(content, encoding="utf-8")


def parse_num_list(raw: str, cast: type) -> list[Any]:
    values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Candidate list is empty.")
    return values


def format_lr_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_cuda_error(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return "cuda error" in lowered or "device-side assert" in lowered


# ==============================================================
# RESULT PARSING
# ==============================================================
def read_results_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in (None, ""):
                    row[key] = None
                    continue
                if key == "epoch":
                    row[key] = int(float(value))
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No training rows found in {path}")
    return rows


def find_checkpoint_row(rows: list[dict[str, Any]], checkpoint: int) -> dict[str, Any]:
    exact = [row for row in rows if row["epoch"] == checkpoint]
    if exact:
        return exact[0]

    earlier = [row for row in rows if row["epoch"] <= checkpoint]
    if earlier:
        return earlier[-1]
    return rows[-1]


def calc_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def metric_row(
    row: dict[str, Any],
    lr0: float,
    batch: int,
    label: str,
    checkpoint: int,
    run_dir: Path,
    weight_path: Path,
    source: str = "new_experiment",
) -> dict[str, Any]:
    precision = row["metrics/precision(B)"]
    recall = row["metrics/recall(B)"]
    train_loss = row["train/box_loss"] + row["train/cls_loss"] + row["train/dfl_loss"]
    val_loss = row["val/box_loss"] + row["val/cls_loss"] + row["val/dfl_loss"]

    return {
        "source": source,
        "config": label,
        "lr0": lr0,
        "batch": batch,
        "checkpoint": checkpoint,
        "actual_epoch": row["epoch"],
        "precision": precision,
        "recall": recall,
        "f1": calc_f1(precision, recall),
        "map50": row["metrics/mAP50(B)"],
        "map50_95": row["metrics/mAP50-95(B)"],
        "train_box_loss": row["train/box_loss"],
        "train_cls_loss": row["train/cls_loss"],
        "train_dfl_loss": row["train/dfl_loss"],
        "train_loss": train_loss,
        "val_box_loss": row["val/box_loss"],
        "val_cls_loss": row["val/cls_loss"],
        "val_dfl_loss": row["val/dfl_loss"],
        "val_loss": val_loss,
        "run_dir": str(run_dir),
        "weight_path": str(weight_path),
    }


def summarize_run(run_dir: Path, lr0: float, batch: int, label: str, source: str = "new_experiment") -> dict[str, Any]:
    rows = read_results_csv(run_dir / "results.csv")
    best_row = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
    final_row = rows[-1]
    weight_path = run_dir / "weights" / "best.pt"

    checkpoints = [
        metric_row(find_checkpoint_row(rows, checkpoint), lr0, batch, label, checkpoint, run_dir, weight_path, source)
        for checkpoint in CHECKPOINTS
    ]

    return {
        "status": "ok",
        "source": source,
        "config": label,
        "lr0": lr0,
        "batch": batch,
        "max_epochs": MAX_EPOCHS,
        "best_epoch": best_row["epoch"],
        "best_precision": best_row["metrics/precision(B)"],
        "best_recall": best_row["metrics/recall(B)"],
        "best_f1": calc_f1(best_row["metrics/precision(B)"], best_row["metrics/recall(B)"]),
        "best_map50": best_row["metrics/mAP50(B)"],
        "best_map50_95": best_row["metrics/mAP50-95(B)"],
        "final_epoch": final_row["epoch"],
        "final_precision": final_row["metrics/precision(B)"],
        "final_recall": final_row["metrics/recall(B)"],
        "final_f1": calc_f1(final_row["metrics/precision(B)"], final_row["metrics/recall(B)"]),
        "final_map50": final_row["metrics/mAP50(B)"],
        "final_map50_95": final_row["metrics/mAP50-95(B)"],
        "duration_seconds": final_row["time"],
        "curve_epochs": [row["epoch"] for row in rows],
        "curve_train_loss": [row["train/box_loss"] + row["train/cls_loss"] + row["train/dfl_loss"] for row in rows],
        "curve_val_loss": [row["val/box_loss"] + row["val/cls_loss"] + row["val/dfl_loss"] for row in rows],
        "curve_map50_95": [row["metrics/mAP50-95(B)"] for row in rows],
        "curve_map50": [row["metrics/mAP50(B)"] for row in rows],
        "curve_precision": [row["metrics/precision(B)"] for row in rows],
        "curve_recall": [row["metrics/recall(B)"] for row in rows],
        "checkpoints": checkpoints,
        "run_dir": str(run_dir),
        "weight_path": str(weight_path),
    }


def summarize_previous_baseline() -> dict[str, Any] | None:
    if not (PREVIOUS_RUN_DIR / "results.csv").exists() or not PREVIOUS_WEIGHT.exists():
        return None
    return summarize_run(
        PREVIOUS_RUN_DIR,
        lr0=0.01,
        batch=16,
        label="previous best.pt baseline",
        source="previous_weight_baseline",
    )


def best_score(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        result["best_map50_95"],
        result["best_map50"],
        result["best_recall"],
        result["best_precision"],
    )


# ==============================================================
# TRAINING
# ==============================================================
def build_run_name(lr0: float, batch: int, suffix: str = "") -> str:
    name = f"lr{format_lr_tag(lr0)}_b{batch}_e{MAX_EPOCHS}"
    return f"{name}_{suffix}" if suffix else name


def train_one_config(
    lr0: float,
    batch: int,
    model_path: Path,
    project_dir: Path,
    device: str,
    imgsz: int,
    seed: int,
    workers: int,
    lrf: float,
    optimizer: str,
    run_suffix: str = "",
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
        raise

    label = f"LR={lr0} B={batch}"
    run_name = build_run_name(lr0, batch, run_suffix)

    print("\n" + "=" * 78)
    print(f"{label} | epochs={MAX_EPOCHS} | optimizer={optimizer} | device={device if device else 'auto'}")
    print("=" * 78)

    model = YOLO(str(model_path))
    results = model.train(
        data=str(DATA_YAML),
        epochs=MAX_EPOCHS,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        lrf=lrf,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
        patience=MAX_EPOCHS,
        optimizer=optimizer,
        seed=seed,
        deterministic=True,
        workers=workers,
        pretrained=True,
        verbose=True,
        save=True,
        plots=True,
    )

    summary = summarize_run(Path(results.save_dir), lr0=lr0, batch=batch, label=label)
    summary["actual_device"] = device if device else "auto"
    summary["optimizer"] = optimizer
    return summary


def child_command(
    script_path: Path,
    lr0: float,
    batch: int,
    args: argparse.Namespace,
    project_dir: Path,
    summary_path: Path,
    run_suffix: str = "",
) -> list[str]:
    command = [
        sys.executable,
        str(script_path),
        "--child-run",
        "--child-lr0",
        str(lr0),
        "--child-batch",
        str(batch),
        "--child-project-dir",
        str(project_dir),
        "--child-summary-path",
        str(summary_path),
        "--model",
        str(args.model),
        "--device",
        str(args.device),
        "--imgsz",
        str(args.imgsz),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--lrf",
        str(args.lrf),
        "--optimizer",
        str(args.optimizer),
        "--max-epochs",
        str(args.max_epochs),
        "--checkpoints",
        str(args.checkpoints),
    ]
    if run_suffix:
        command.extend(["--child-run-suffix", run_suffix])
    return command


def run_child(lr0: float, batch: int, args: argparse.Namespace, project_dir: Path, run_suffix: str = "") -> dict[str, Any]:
    summary_dir = project_dir / "_child_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{build_run_name(lr0, batch, run_suffix)}.json"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = child_command(Path(__file__).resolve(), lr0, batch, args, project_dir, summary_path, run_suffix)

    completed = subprocess.run(command, cwd=str(BASE_DIR), env=env, check=False)
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "status": "failed",
        "config": f"LR={lr0} B={batch}",
        "lr0": lr0,
        "batch": batch,
        "error": f"Child process exited with code {completed.returncode}.",
    }


def execute_config(lr0: float, batch: int, args: argparse.Namespace, project_dir: Path) -> dict[str, Any]:
    result = run_child(lr0, batch, args, project_dir)
    if result.get("status") == "ok":
        return result

    if args.retry_cpu_on_cuda_error and is_cuda_error(result.get("error")):
        print("CUDA failed for this config. Retrying on CPU.")
        retry_args = argparse.Namespace(**vars(args))
        retry_args.device = "cpu"
        retry = run_child(lr0, batch, retry_args, project_dir, run_suffix="cpu_retry")
        retry["retried_from_cuda_error"] = True
        return retry

    return result


# ==============================================================
# PLOTS
# ==============================================================
def plot_curves(results: list[dict[str, Any]], baseline: dict[str, Any] | None, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    if baseline:
        ax.plot(
            baseline["curve_epochs"],
            baseline["curve_map50_95"],
            color="#444444",
            linestyle="--",
            linewidth=2.0,
            label="previous best.pt baseline",
        )

    for result in results:
        if result.get("status") != "ok":
            continue
        ax.plot(result["curve_epochs"], result["curve_map50_95"], linewidth=2.0, label=result["config"])
        best_idx = int(np.argmax(result["curve_map50_95"]))
        ax.scatter(result["curve_epochs"][best_idx], result["curve_map50_95"][best_idx], s=45)

    for checkpoint in CHECKPOINTS:
        ax.axvline(checkpoint, color="#999999", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_title("YOLOv8 LR x Batch Comparison: mAP50-95 Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_checkpoint_metric(checkpoint_rows: list[dict[str, Any]], metric: str, title: str, output_path: Path) -> None:
    configs = sorted({row["config"] for row in checkpoint_rows if row["source"] == "new_experiment"})
    x = np.arange(len(configs))
    width = 0.18
    offsets = np.linspace(-0.27, 0.27, len(CHECKPOINTS))

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2E86DE", "#28B463", "#F39C12", "#C0392B"]

    for offset, checkpoint, color in zip(offsets, CHECKPOINTS, colors):
        values = []
        for config in configs:
            match = [
                row for row in checkpoint_rows
                if row["config"] == config and row["checkpoint"] == checkpoint and row["source"] == "new_experiment"
            ]
            values.append(match[0][metric] if match else np.nan)
        bars = ax.bar(x + offset, values, width, label=f"Epoch {checkpoint}", color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lr_curves_grid(results: list[dict[str, Any]], output_path: Path) -> None:
    successful = [result for result in results if result.get("status") == "ok"]
    batches = sorted({result["batch"] for result in successful})
    if not batches:
        return

    fig, axes = plt.subplots(2, len(batches), figsize=(7 * len(batches), 9), squeeze=False)
    colors = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6", "#16A085"]

    for col, batch in enumerate(batches):
        batch_results = sorted(
            [result for result in successful if result["batch"] == batch],
            key=lambda result: result["lr0"],
        )

        ax_loss = axes[0][col]
        ax_map = axes[1][col]
        for idx, result in enumerate(batch_results):
            color = colors[idx % len(colors)]
            label = f"LR={result['lr0']}"
            ax_loss.plot(result["curve_epochs"], result["curve_val_loss"], label=label, color=color, linewidth=1.8)
            ax_map.plot(result["curve_epochs"], result["curve_map50_95"], label=label, color=color, linewidth=1.8)

        ax_loss.set_title(f"Batch={batch} - Validation Loss", fontsize=11, fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(fontsize=9)

        for checkpoint in CHECKPOINTS:
            ax_map.axvline(checkpoint, color="#999999", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_map.set_title(f"Batch={batch} - mAP50-95", fontsize=11, fontweight="bold")
        ax_map.set_xlabel("Epoch")
        ax_map.set_ylabel("mAP50-95")
        ax_map.grid(True, alpha=0.3)
        ax_map.legend(fontsize=9)

    plt.suptitle("Learning Rate Comparison by Batch", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_checkpoint_effect_grid(checkpoint_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in checkpoint_rows if row["source"] == "new_experiment"]
    configs = sorted({row["config"] for row in rows})
    if not configs:
        return

    metric_specs = [
        ("map50_95", "mAP50-95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
    ]
    colors = ["#F39C12", "#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"]
    x = np.arange(len(configs))
    width = 0.75 / max(1, len(CHECKPOINTS))
    offsets = np.linspace(-0.35, 0.35, len(CHECKPOINTS))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (metric, title) in zip(axes.ravel(), metric_specs):
        for offset, checkpoint, color in zip(offsets, CHECKPOINTS, colors):
            values = []
            for config in configs:
                match = [
                    row for row in rows
                    if row["config"] == config and row["checkpoint"] == checkpoint
                ]
                values.append(match[0][metric] * 100 if match else np.nan)
            bars = ax.bar(x + offset, values, width, label=f"Epoch {checkpoint}", color=color, alpha=0.9)
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.4,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_ylim(0, 105)
        ax.set_title(f"{title} @ Checkpoints", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle("Effect of Epoch Checkpoints (All Configs)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_metrics(checkpoint_rows: list[dict[str, Any]], output_path: Path) -> None:
    final_rows = [
        row for row in checkpoint_rows
        if row["checkpoint"] == max(CHECKPOINTS) and row["source"] == "new_experiment"
    ]
    configs = [row["config"] for row in final_rows]
    x = np.arange(len(configs))
    width = 0.22

    fig, ax = plt.subplots(figsize=(13, 6))
    metrics = [
        ("precision", "#2E86DE", "Precision"),
        ("recall", "#28B463", "Recall"),
        ("f1", "#E74C3C", "F1-Score"),
    ]

    for idx, (metric, color, label) in enumerate(metrics):
        values = [row[metric] * 100 for row in final_rows]
        bars = ax.bar(x + (idx - 1) * width, values, width, label=label, color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 105)
    ax.set_title(f"Precision / Recall / F1 @ Epoch {max(CHECKPOINTS)} - All Configs", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_map_summary(results: list[dict[str, Any]], output_path: Path) -> None:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        return

    configs = [result["config"] for result in successful]
    best_values = [result["best_map50_95"] * 100 for result in successful]
    final_values = [result["final_map50_95"] * 100 for result in successful]
    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width / 2, best_values, width, label="Best mAP50-95", color="#2E86DE", alpha=0.9)
    b2 = ax.bar(x + width / 2, final_values, width, label=f"mAP50-95 @ Epoch {MAX_EPOCHS}", color="#F39C12", alpha=0.9)

    for bars in [b1, b2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 105)
    ax.set_title("Best vs Final mAP50-95", fontsize=13, fontweight="bold")
    ax.set_ylabel("mAP50-95 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


# ==============================================================
# REPORTING
# ==============================================================
def flatten_run_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "source": result.get("source"),
        "config": result.get("config"),
        "lr0": result.get("lr0"),
        "batch": result.get("batch"),
        "best_epoch": result.get("best_epoch"),
        "best_precision": result.get("best_precision"),
        "best_recall": result.get("best_recall"),
        "best_f1": result.get("best_f1"),
        "best_map50": result.get("best_map50"),
        "best_map50_95": result.get("best_map50_95"),
        "final_epoch": result.get("final_epoch"),
        "final_precision": result.get("final_precision"),
        "final_recall": result.get("final_recall"),
        "final_f1": result.get("final_f1"),
        "final_map50": result.get("final_map50"),
        "final_map50_95": result.get("final_map50_95"),
        "duration_seconds": result.get("duration_seconds"),
        "actual_device": result.get("actual_device"),
        "optimizer": result.get("optimizer"),
        "retried_from_cuda_error": result.get("retried_from_cuda_error", False),
        "weight_path": result.get("weight_path"),
        "run_dir": result.get("run_dir"),
        "error": result.get("error"),
    }


def write_report(path: Path, baseline: dict[str, Any] | None, results: list[dict[str, Any]]) -> None:
    successful = [result for result in results if result.get("status") == "ok"]
    best = max(successful, key=best_score) if successful else None

    lines = [
        "YOLOv8 Bottle Cap Hyperparameter Checkpoint Report",
        "=" * 58,
        "",
        f"Search space: lr0={LR_CANDIDATES}, batch={BATCH_CANDIDATES}, epochs={MAX_EPOCHS}",
        f"Checkpoints: {CHECKPOINTS}",
        f"Optimizer: {OPTIMIZER}",
        "YOLO augmentation: default Ultralytics training augmentation",
        "",
    ]

    if baseline:
        lines.extend(
            [
                "Previous trained weight baseline",
                f"- weight: {PREVIOUS_WEIGHT}",
                f"- best: epoch={baseline['best_epoch']}, mAP50-95={baseline['best_map50_95']:.4f}, mAP50={baseline['best_map50']:.4f}",
                "",
            ]
        )

    if best:
        lines.extend(
            [
                "Best new experiment",
                f"- config: {best['config']}",
                f"- lr0={best['lr0']}, batch={best['batch']}",
                f"- best epoch={best['best_epoch']}",
                f"- best mAP50-95={best['best_map50_95']:.4f}, mAP50={best['best_map50']:.4f}",
                f"- weight: {best['weight_path']}",
                "",
            ]
        )

    lines.append("Run summary")
    for result in results:
        if result.get("status") != "ok":
            lines.append(f"- {result.get('config')}: FAILED ({result.get('error')})")
            continue
        lines.append(
            f"- {result['config']}: best mAP50-95={result['best_map50_95']:.4f} "
            f"at epoch {result['best_epoch']}, final mAP50-95={result['final_map50_95']:.4f}, "
            f"device={result.get('actual_device')}"
        )

    if best:
        lines.extend(
            [
                "",
                "Short presentation summary",
                f"- Selected config by best mAP50-95: {best['config']}",
                f"- Best epoch from training: {best['best_epoch']}",
                f"- Best mAP50-95: {best['best_map50_95']:.4f}",
                f"- Precision/Recall/F1 at best epoch: {best['best_precision']:.4f} / {best['best_recall']:.4f} / {best['best_f1']:.4f}",
                "- mAP50-95 is used as the main metric because mAP50 is often saturated and less sensitive to box quality.",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ==============================================================
# CLI
# ==============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 LR x batch comparison with checkpoint metrics.")
    parser.add_argument("--model", default=str(YOLO_MODEL))
    parser.add_argument("--device", default="")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--lrf", type=float, default=LRF)
    parser.add_argument("--optimizer", default=OPTIMIZER)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--checkpoints", default="50,100,150")
    parser.add_argument("--lrs", default="0.001,0.003,0.005")
    parser.add_argument("--batches", default="8")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--include-previous-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-cpu-on-cuda-error", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--child-lr0", type=float, default=0.0)
    parser.add_argument("--child-batch", type=int, default=0)
    parser.add_argument("--child-project-dir", default="")
    parser.add_argument("--child-summary-path", default="")
    parser.add_argument("--child-run-suffix", default="")
    return parser.parse_args()


def child_main(args: argparse.Namespace) -> int:
    global CHECKPOINTS, MAX_EPOCHS, OPTIMIZER

    CHECKPOINTS = parse_num_list(args.checkpoints, int)
    MAX_EPOCHS = args.max_epochs
    OPTIMIZER = args.optimizer

    ensure_dataset()
    write_data_yaml()

    summary_path = Path(args.child_summary_path)
    try:
        result = train_one_config(
            lr0=args.child_lr0,
            batch=args.child_batch,
            model_path=Path(args.model),
            project_dir=Path(args.child_project_dir),
            device=args.device,
            imgsz=args.imgsz,
            seed=args.seed,
            workers=args.workers,
            lrf=args.lrf,
            optimizer=args.optimizer,
            run_suffix=args.child_run_suffix,
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "failed",
            "config": f"LR={args.child_lr0} B={args.child_batch}",
            "lr0": args.child_lr0,
            "batch": args.child_batch,
            "actual_device": args.device or "auto",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[WARNING] Run failed: {exc}")
        print(result["traceback"])
        write_json(summary_path, result)
        return 1

    write_json(summary_path, result)
    return 0


def main() -> None:
    global BATCH_CANDIDATES, CHECKPOINTS, LR_CANDIDATES, MAX_EPOCHS, OPTIMIZER

    args = parse_args()
    LR_CANDIDATES = parse_num_list(args.lrs, float)
    BATCH_CANDIDATES = parse_num_list(args.batches, int)
    CHECKPOINTS = parse_num_list(args.checkpoints, int)
    MAX_EPOCHS = args.max_epochs
    OPTIMIZER = args.optimizer

    if args.child_run:
        raise SystemExit(child_main(args))

    ensure_dataset()
    write_data_yaml()

    lrs = LR_CANDIDATES
    batches = BATCH_CANDIDATES

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path(args.output_dir) / f"checkpoint_compare_{timestamp}"
    project_dir.mkdir(parents=True, exist_ok=False)

    print("\n" + "=" * 78)
    print("YOLOv8 Bottle Cap LR x Batch Checkpoint Comparison")
    print("=" * 78)
    print(f"Dataset    : {DATASET_DIR}")
    print(f"Model      : {args.model}")
    print(f"LRs        : {lrs}")
    print(f"Batches    : {batches}")
    print(f"Epochs/run : {MAX_EPOCHS}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Optimizer  : {OPTIMIZER}")
    print(f"Output     : {project_dir}")

    baseline = summarize_previous_baseline() if args.include_previous_baseline else None
    if baseline:
        print(
            f"Previous baseline: best mAP50-95={baseline['best_map50_95']:.4f} "
            f"at epoch {baseline['best_epoch']}"
        )

    results: list[dict[str, Any]] = []
    total = len(lrs) * len(batches)
    index = 0
    for batch in batches:
        for lr0 in lrs:
            index += 1
            print(f"\nRunning experiment {index}/{total}: LR={lr0} B={batch}")
            result = execute_config(lr0, batch, args, project_dir)
            results.append(result)
            write_json(project_dir / "progress.json", {"baseline": baseline, "results": results})

            if result.get("status") == "ok":
                print(
                    f"Result: best mAP50-95={result['best_map50_95']:.4f} "
                    f"at epoch {result['best_epoch']}"
                )
            else:
                print(f"[WARNING] Failed: {result.get('error')}")

    checkpoint_rows = []
    if baseline:
        checkpoint_rows.extend(baseline["checkpoints"])
    for result in results:
        if result.get("status") == "ok":
            checkpoint_rows.extend(result["checkpoints"])

    all_for_run_csv = ([baseline] if baseline else []) + results
    write_csv(project_dir / "run_summary.csv", [flatten_run_summary(result) for result in all_for_run_csv])
    write_csv(project_dir / "checkpoint_summary.csv", checkpoint_rows)
    write_json(project_dir / "full_summary.json", {"baseline": baseline, "results": results, "checkpoints": checkpoint_rows})
    write_report(project_dir / "report.txt", baseline, results)

    plot_curves(results, baseline, project_dir / "map50_95_curves.png")
    plot_lr_curves_grid(results, project_dir / "lr_loss_map_curves.png")
    plot_checkpoint_metric(checkpoint_rows, "map50_95", "mAP50-95 at Checkpoints", project_dir / "checkpoint_map50_95.png")
    plot_checkpoint_metric(checkpoint_rows, "map50", "mAP50 at Checkpoints", project_dir / "checkpoint_map50.png")
    plot_checkpoint_effect_grid(checkpoint_rows, project_dir / "checkpoint_effect_metrics.png")
    plot_best_map_summary(results, project_dir / "best_vs_final_map50_95.png")
    final_metrics_name = f"final_metrics_epoch{max(CHECKPOINTS)}.png"
    plot_final_metrics(checkpoint_rows, project_dir / final_metrics_name)

    successful = [result for result in results if result.get("status") == "ok"]
    best = max(successful, key=best_score) if successful else None

    print("\n" + "=" * 78)
    print("Comparison complete")
    print("=" * 78)
    if best:
        print(f"Best config : {best['config']}")
        print(f"Best epoch  : {best['best_epoch']}")
        print(f"Best mAP50-95: {best['best_map50_95']:.4f}")
        print(f"Best weight : {best['weight_path']}")
    print(f"Outputs     : {project_dir}")
    print("  - run_summary.csv")
    print("  - checkpoint_summary.csv")
    print("  - report.txt")
    print("  - map50_95_curves.png")
    print("  - lr_loss_map_curves.png")
    print("  - checkpoint_map50_95.png")
    print("  - checkpoint_map50.png")
    print("  - checkpoint_effect_metrics.png")
    print("  - best_vs_final_map50_95.png")
    print(f"  - {final_metrics_name}")


if __name__ == "__main__":
    main()
