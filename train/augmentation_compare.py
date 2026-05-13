#!/usr/bin/env python3
"""
YOLOv8 augmentation comparison for bottle cap detection.

This script keeps the selected training settings fixed:
- batch = 8
- lr0 = 0.005
- optimizer = SGD
- epochs = 150

Then it compares several YOLO augmentation policies:
- No Aug
- Default Aug
- Light Aug
- No Mosaic
- Strong Color/Scale

Outputs include CSV summaries, a text report, and multiple comparison graphs.
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

OUTPUT_DIR = TRAINRUN_DIR / "yolo_augmentation_output"

CLASS_NAMES = ["closed_cap", "half_cap", "no_cap"]
CHECKPOINTS = [50, 100, 150]
MAX_EPOCHS = 150
IMAGE_SIZE = 640
BATCH = 8
LR0 = 0.005
LRF = 0.01
OPTIMIZER = "SGD"
SEED = 42
WORKERS = 8


AUG_POLICIES: dict[str, dict[str, Any]] = {
    "no_aug": {
        "label": "No Aug",
        "description": "Disable main augmentation parameters.",
        "params": {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "auto_augment": None,
        },
    },
    "default_aug": {
        "label": "Default Aug",
        "description": "Use Ultralytics YOLO default augmentation.",
        "params": {},
    },
    "light_aug": {
        "label": "Light Aug",
        "description": "Use gentler color, scale, translate, mosaic, and erasing.",
        "params": {
            "hsv_h": 0.01,
            "hsv_s": 0.3,
            "hsv_v": 0.2,
            "translate": 0.05,
            "scale": 0.3,
            "fliplr": 0.5,
            "mosaic": 0.5,
            "mixup": 0.0,
            "cutmix": 0.0,
            "erasing": 0.1,
        },
    },
    "no_mosaic": {
        "label": "No Mosaic",
        "description": "Keep YOLO defaults but disable mosaic.",
        "params": {
            "mosaic": 0.0,
        },
    },
    "strong_color_scale": {
        "label": "Strong Color/Scale",
        "description": "Use stronger color, scale, translate, mosaic, and erasing.",
        "params": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mosaic": 0.5,
            "mixup": 0.0,
            "cutmix": 0.0,
            "erasing": 0.2,
        },
    },
}


# ==============================================================
# HELPERS
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


def parse_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Policy list is empty.")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Checkpoint list is empty.")
    return values


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


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


def calc_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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


def metric_row(
    row: dict[str, Any],
    policy_key: str,
    policy_label: str,
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
        "policy_key": policy_key,
        "policy": policy_label,
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


def summarize_run(
    run_dir: Path,
    policy_key: str,
    policy_label: str,
    source: str = "new_experiment",
) -> dict[str, Any]:
    rows = read_results_csv(run_dir / "results.csv")
    best_row = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
    final_row = rows[-1]
    weight_path = run_dir / "weights" / "best.pt"
    checkpoints = [
        metric_row(find_checkpoint_row(rows, checkpoint), policy_key, policy_label, checkpoint, run_dir, weight_path, source)
        for checkpoint in CHECKPOINTS
    ]

    return {
        "status": "ok",
        "source": source,
        "policy_key": policy_key,
        "policy": policy_label,
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
        policy_key="previous_baseline",
        policy_label="Previous best.pt",
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
def train_one_policy(
    policy_key: str,
    policy: dict[str, Any],
    model_path: Path,
    project_dir: Path,
    device: str,
    imgsz: int,
    epochs: int,
    batch: int,
    lr0: float,
    lrf: float,
    optimizer: str,
    seed: int,
    workers: int,
    run_suffix: str = "",
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
        raise

    run_name = safe_name(policy["label"])
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"

    print("\n" + "=" * 78)
    print(f"{policy['label']} | epochs={epochs} batch={batch} lr0={lr0} optimizer={optimizer}")
    print(f"Device: {device if device else 'auto'}")
    print("=" * 78)

    train_kwargs = {
        "data": str(DATA_YAML),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "lr0": lr0,
        "lrf": lrf,
        "device": device,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": False,
        "patience": epochs,
        "optimizer": optimizer,
        "seed": seed,
        "deterministic": True,
        "workers": workers,
        "pretrained": True,
        "verbose": True,
        "save": True,
        "plots": True,
    }
    train_kwargs.update(policy["params"])

    model = YOLO(str(model_path))
    results = model.train(**train_kwargs)

    summary = summarize_run(Path(results.save_dir), policy_key, policy["label"])
    summary["actual_device"] = device if device else "auto"
    summary["optimizer"] = optimizer
    summary["augmentation_params"] = policy["params"]
    return summary


def child_command(
    script_path: Path,
    policy_key: str,
    args: argparse.Namespace,
    project_dir: Path,
    summary_path: Path,
    run_suffix: str = "",
) -> list[str]:
    command = [
        sys.executable,
        str(script_path),
        "--child-run",
        "--child-policy",
        policy_key,
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
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--lr0",
        str(args.lr0),
        "--lrf",
        str(args.lrf),
        "--optimizer",
        str(args.optimizer),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--checkpoints",
        str(args.checkpoints),
    ]
    if run_suffix:
        command.extend(["--child-run-suffix", run_suffix])
    return command


def run_child(policy_key: str, args: argparse.Namespace, project_dir: Path, run_suffix: str = "") -> dict[str, Any]:
    summary_dir = project_dir / "_child_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{safe_name(AUG_POLICIES[policy_key]['label'])}{'_' + run_suffix if run_suffix else ''}.json"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = child_command(Path(__file__).resolve(), policy_key, args, project_dir, summary_path, run_suffix)
    completed = subprocess.run(command, cwd=str(BASE_DIR), env=env, check=False)

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "status": "failed",
        "policy_key": policy_key,
        "policy": AUG_POLICIES[policy_key]["label"],
        "error": f"Child process exited with code {completed.returncode}.",
    }


def execute_policy(policy_key: str, args: argparse.Namespace, project_dir: Path) -> dict[str, Any]:
    result = run_child(policy_key, args, project_dir)
    if result.get("status") == "ok":
        return result

    if args.retry_cpu_on_cuda_error and is_cuda_error(result.get("error")):
        print("CUDA failed for this policy. Retrying on CPU.")
        retry_args = argparse.Namespace(**vars(args))
        retry_args.device = "cpu"
        retry = run_child(policy_key, retry_args, project_dir, run_suffix="cpu_retry")
        retry["retried_from_cuda_error"] = True
        return retry

    return result


# ==============================================================
# PLOTS
# ==============================================================
def plot_map_curves(results: list[dict[str, Any]], baseline: dict[str, Any] | None, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    if baseline:
        ax.plot(
            baseline["curve_epochs"],
            baseline["curve_map50_95"],
            color="#444444",
            linestyle="--",
            linewidth=2.0,
            label="Previous best.pt",
        )

    for result in results:
        if result.get("status") != "ok":
            continue
        ax.plot(result["curve_epochs"], result["curve_map50_95"], linewidth=2.0, label=result["policy"])
        best_idx = int(np.argmax(result["curve_map50_95"]))
        ax.scatter(result["curve_epochs"][best_idx], result["curve_map50_95"][best_idx], s=45)

    for checkpoint in CHECKPOINTS:
        ax.axvline(checkpoint, color="#999999", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_title("Augmentation Comparison: mAP50-95 Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_map_grid(results: list[dict[str, Any]], output_path: Path) -> None:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    colors = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6", "#16A085"]

    for idx, result in enumerate(successful):
        color = colors[idx % len(colors)]
        axes[0].plot(result["curve_epochs"], result["curve_val_loss"], label=result["policy"], color=color, linewidth=1.8)
        axes[1].plot(result["curve_epochs"], result["curve_map50_95"], label=result["policy"], color=color, linewidth=1.8)

    axes[0].set_title("Validation Loss by Augmentation Policy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_title("mAP50-95 by Augmentation Policy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP50-95")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.suptitle("Augmentation Loss and Detection Metric Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_checkpoint_metric(checkpoint_rows: list[dict[str, Any]], metric: str, title: str, output_path: Path) -> None:
    rows = [row for row in checkpoint_rows if row["source"] == "new_experiment"]
    policies = [row["policy"] for row in rows if row["checkpoint"] == max(CHECKPOINTS)]
    if not policies:
        policies = sorted({row["policy"] for row in rows})
    x = np.arange(len(policies))
    width = 0.75 / max(1, len(CHECKPOINTS))
    offsets = np.linspace(-0.35, 0.35, len(CHECKPOINTS))
    colors = ["#F39C12", "#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for offset, checkpoint, color in zip(offsets, CHECKPOINTS, colors):
        values = []
        for policy in policies:
            match = [row for row in rows if row["policy"] == policy and row["checkpoint"] == checkpoint]
            values.append(match[0][metric] if match else np.nan)
        bars = ax.bar(x + offset, values, width, label=f"Epoch {checkpoint}", color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_checkpoint_effect_grid(checkpoint_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in checkpoint_rows if row["source"] == "new_experiment"]
    policies = [row["policy"] for row in rows if row["checkpoint"] == max(CHECKPOINTS)]
    if not policies:
        policies = sorted({row["policy"] for row in rows})
    if not policies:
        return

    metric_specs = [
        ("map50_95", "mAP50-95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
    ]
    colors = ["#F39C12", "#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"]
    x = np.arange(len(policies))
    width = 0.75 / max(1, len(CHECKPOINTS))
    offsets = np.linspace(-0.35, 0.35, len(CHECKPOINTS))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (metric, title) in zip(axes.ravel(), metric_specs):
        for offset, checkpoint, color in zip(offsets, CHECKPOINTS, colors):
            values = []
            for policy in policies:
                match = [row for row in rows if row["policy"] == policy and row["checkpoint"] == checkpoint]
                values.append(match[0][metric] * 100 if match else np.nan)
            bars = ax.bar(x + offset, values, width, label=f"Epoch {checkpoint}", color=color, alpha=0.9)
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylim(0, 105)
        ax.set_title(f"{title} @ Checkpoints", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle("Effect of Augmentation Policies Across Checkpoints", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_metrics(checkpoint_rows: list[dict[str, Any]], output_path: Path) -> None:
    final_rows = [
        row for row in checkpoint_rows
        if row["checkpoint"] == max(CHECKPOINTS) and row["source"] == "new_experiment"
    ]
    if not final_rows:
        return

    policies = [row["policy"] for row in final_rows]
    x = np.arange(len(policies))
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
    ax.set_title(f"Precision / Recall / F1 @ Epoch {max(CHECKPOINTS)} - Augmentation Policies", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_best_vs_final(results: list[dict[str, Any]], output_path: Path) -> None:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        return

    policies = [result["policy"] for result in successful]
    best_values = [result["best_map50_95"] * 100 for result in successful]
    final_values = [result["final_map50_95"] * 100 for result in successful]
    x = np.arange(len(policies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width / 2, best_values, width, label="Best mAP50-95", color="#2E86DE", alpha=0.9)
    b2 = ax.bar(x + width / 2, final_values, width, label=f"mAP50-95 @ Epoch {MAX_EPOCHS}", color="#F39C12", alpha=0.9)

    for bars in [b1, b2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 105)
    ax.set_title("Best vs Final mAP50-95 by Augmentation Policy", fontsize=13, fontweight="bold")
    ax.set_ylabel("mAP50-95 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right")
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
        "policy_key": result.get("policy_key"),
        "policy": result.get("policy"),
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
        "augmentation_params": json.dumps(result.get("augmentation_params", {})),
        "weight_path": result.get("weight_path"),
        "run_dir": result.get("run_dir"),
        "error": result.get("error"),
    }


def write_report(path: Path, baseline: dict[str, Any] | None, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    successful = [result for result in results if result.get("status") == "ok"]
    best = max(successful, key=best_score) if successful else None

    lines = [
        "YOLOv8 Bottle Cap Augmentation Comparison Report",
        "=" * 56,
        "",
        f"Training settings: model={args.model}, batch={args.batch}, lr0={args.lr0}, epochs={MAX_EPOCHS}, optimizer={args.optimizer}",
        f"Checkpoints: {CHECKPOINTS}",
        "",
        "Compared augmentation policies:",
    ]
    for key in args.policy_keys:
        policy = AUG_POLICIES[key]
        lines.append(f"- {policy['label']}: {policy['description']}")
    lines.append("")

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
                "Best augmentation policy",
                f"- policy: {best['policy']}",
                f"- best epoch: {best['best_epoch']}",
                f"- best mAP50-95: {best['best_map50_95']:.4f}",
                f"- mAP50: {best['best_map50']:.4f}",
                f"- precision/recall/F1: {best['best_precision']:.4f} / {best['best_recall']:.4f} / {best['best_f1']:.4f}",
                f"- weight: {best['weight_path']}",
                "",
            ]
        )

    lines.append("Run summary")
    for result in results:
        if result.get("status") != "ok":
            lines.append(f"- {result.get('policy')}: FAILED ({result.get('error')})")
            continue
        lines.append(
            f"- {result['policy']}: best mAP50-95={result['best_map50_95']:.4f} "
            f"at epoch {result['best_epoch']}, final mAP50-95={result['final_map50_95']:.4f}, "
            f"device={result.get('actual_device')}"
        )

    if best:
        lines.extend(
            [
                "",
                "Short presentation summary",
                f"- The selected augmentation setting is {best['policy']} because it achieved the highest best mAP50-95.",
                "- mAP50-95 is used as the main metric because it is stricter than mAP50 and better reflects bounding-box quality.",
                "- Precision, recall, and F1 are included to show whether the model is balanced between false positives and missed detections.",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ==============================================================
# CLI
# ==============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLOv8 augmentation policies for bottle cap detection.")
    parser.add_argument("--model", default=str(YOLO_MODEL))
    parser.add_argument("--device", default="")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr0", type=float, default=LR0)
    parser.add_argument("--lrf", type=float, default=LRF)
    parser.add_argument("--optimizer", default=OPTIMIZER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--checkpoints", default="50,100,150")
    parser.add_argument("--policies", default="no_aug,default_aug,light_aug,no_mosaic,strong_color_scale")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--include-previous-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-cpu-on-cuda-error", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--child-policy", default="")
    parser.add_argument("--child-project-dir", default="")
    parser.add_argument("--child-summary-path", default="")
    parser.add_argument("--child-run-suffix", default="")
    return parser.parse_args()


def child_main(args: argparse.Namespace) -> int:
    global CHECKPOINTS, MAX_EPOCHS

    CHECKPOINTS = parse_int_list(args.checkpoints)
    MAX_EPOCHS = args.epochs
    ensure_dataset()
    write_data_yaml()

    policy_key = args.child_policy
    if policy_key not in AUG_POLICIES:
        raise ValueError(f"Unknown augmentation policy: {policy_key}")

    summary_path = Path(args.child_summary_path)
    try:
        result = train_one_policy(
            policy_key=policy_key,
            policy=AUG_POLICIES[policy_key],
            model_path=Path(args.model),
            project_dir=Path(args.child_project_dir),
            device=args.device,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            lr0=args.lr0,
            lrf=args.lrf,
            optimizer=args.optimizer,
            seed=args.seed,
            workers=args.workers,
            run_suffix=args.child_run_suffix,
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "failed",
            "policy_key": policy_key,
            "policy": AUG_POLICIES.get(policy_key, {}).get("label", policy_key),
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
    global CHECKPOINTS, MAX_EPOCHS

    args = parse_args()
    CHECKPOINTS = parse_int_list(args.checkpoints)
    MAX_EPOCHS = args.epochs

    if args.child_run:
        raise SystemExit(child_main(args))

    ensure_dataset()
    write_data_yaml()

    policy_keys = parse_list(args.policies)
    unknown = [key for key in policy_keys if key not in AUG_POLICIES]
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}. Available: {list(AUG_POLICIES)}")
    args.policy_keys = policy_keys

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path(args.output_dir) / f"aug_compare_{timestamp}"
    project_dir.mkdir(parents=True, exist_ok=False)

    print("\n" + "=" * 78)
    print("YOLOv8 Bottle Cap Augmentation Comparison")
    print("=" * 78)
    print(f"Dataset    : {DATASET_DIR}")
    print(f"Model      : {args.model}")
    print(f"Batch      : {args.batch}")
    print(f"LR0        : {args.lr0}")
    print(f"Epochs/run : {args.epochs}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Optimizer  : {args.optimizer}")
    print(f"Policies   : {[AUG_POLICIES[key]['label'] for key in policy_keys]}")
    print(f"Output     : {project_dir}")

    baseline = summarize_previous_baseline() if args.include_previous_baseline else None
    if baseline:
        print(f"Previous baseline: best mAP50-95={baseline['best_map50_95']:.4f} at epoch {baseline['best_epoch']}")

    results: list[dict[str, Any]] = []
    for index, policy_key in enumerate(policy_keys, start=1):
        print(f"\nRunning augmentation experiment {index}/{len(policy_keys)}: {AUG_POLICIES[policy_key]['label']}")
        result = execute_policy(policy_key, args, project_dir)
        results.append(result)
        write_json(project_dir / "progress.json", {"baseline": baseline, "results": results})

        if result.get("status") == "ok":
            print(f"Result: best mAP50-95={result['best_map50_95']:.4f} at epoch {result['best_epoch']}")
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
    write_report(project_dir / "report.txt", baseline, results, args)

    plot_map_curves(results, baseline, project_dir / "map50_95_curves.png")
    plot_loss_map_grid(results, project_dir / "loss_map_curves.png")
    plot_checkpoint_metric(checkpoint_rows, "map50_95", "mAP50-95 at Checkpoints", project_dir / "checkpoint_map50_95.png")
    plot_checkpoint_metric(checkpoint_rows, "map50", "mAP50 at Checkpoints", project_dir / "checkpoint_map50.png")
    plot_checkpoint_effect_grid(checkpoint_rows, project_dir / "checkpoint_effect_metrics.png")
    final_metrics_name = f"final_metrics_epoch{max(CHECKPOINTS)}.png"
    plot_final_metrics(checkpoint_rows, project_dir / final_metrics_name)
    plot_best_vs_final(results, project_dir / "best_vs_final_map50_95.png")

    successful = [result for result in results if result.get("status") == "ok"]
    best = max(successful, key=best_score) if successful else None

    print("\n" + "=" * 78)
    print("Augmentation comparison complete")
    print("=" * 78)
    if best:
        print(f"Best policy : {best['policy']}")
        print(f"Best epoch  : {best['best_epoch']}")
        print(f"Best mAP50-95: {best['best_map50_95']:.4f}")
        print(f"Best weight : {best['weight_path']}")
    print(f"Outputs     : {project_dir}")
    print("  - run_summary.csv")
    print("  - checkpoint_summary.csv")
    print("  - report.txt")
    print("  - map50_95_curves.png")
    print("  - loss_map_curves.png")
    print("  - checkpoint_map50_95.png")
    print("  - checkpoint_map50.png")
    print("  - checkpoint_effect_metrics.png")
    print(f"  - {final_metrics_name}")
    print("  - best_vs_final_map50_95.png")


if __name__ == "__main__":
    main()
