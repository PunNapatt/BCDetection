#!/usr/bin/env python3
"""
Test every augmentation-experiment weight on the same image set.

Outputs:
- One prediction-image folder per augmentation policy
- predictions_long.csv: one row per image per policy
- predictions_review.csv: one row per image, with every policy prediction side by side
- summary.csv: detection/accuracy summary per policy

Accuracy is computed automatically when ground truth is available. Ground truth
can come from either:
1. A source folder with class subfolders, e.g. test/closed_cap/*.jpg
2. A CSV passed with --truth-csv containing image,actual_class columns
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUG_ROOT = SCRIPT_DIR / "yolo_augmentation_output" / "aug_compare_20260512_181523"
DEFAULT_SOURCE = PROJECT_ROOT / "DataPreparation" / "For_Test"
DEFAULT_OUTPUT = SCRIPT_DIR / "augmentation_weight_test_results"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "closed_cap", 1: "half_cap", 2: "no_cap"}
VALID_CLASSES = set(CLASS_NAMES.values())
POLICY_ORDER = ["no_aug", "default_aug", "light_aug", "no_mosaic", "strong_color_scale"]


def find_weights(weights_root: Path) -> dict[str, Path]:
    weights: dict[str, Path] = {}
    for policy in POLICY_ORDER:
        weight = weights_root / policy / "weights" / "best.pt"
        if weight.exists():
            weights[policy] = weight

    missing = [policy for policy in POLICY_ORDER if policy not in weights]
    if missing:
        print(f"[WARNING] Missing weights for: {', '.join(missing)}")

    if not weights:
        print(f"[ERROR] No best.pt files found under: {weights_root}")
        sys.exit(1)

    return weights


def collect_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMG_EXTS:
        return [source]
    if not source.exists():
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)

    images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[ERROR] No images found in: {source}")
        sys.exit(1)
    return images


def infer_truth_from_folder(image_path: Path, source: Path) -> str:
    try:
        rel = image_path.relative_to(source)
    except ValueError:
        return ""

    if len(rel.parts) >= 2 and rel.parts[0] in VALID_CLASSES:
        return rel.parts[0]
    return ""


def load_truth_csv(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        print(f"[ERROR] truth CSV not found: {path}")
        sys.exit(1)

    truth: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "actual_class"}
        if not required.issubset(reader.fieldnames or set()):
            print("[ERROR] truth CSV must contain columns: image,actual_class")
            sys.exit(1)
        for row in reader:
            image = (row.get("image") or "").strip()
            actual_class = (row.get("actual_class") or "").strip()
            if image and actual_class:
                truth[image] = actual_class
                truth[Path(image).name] = actual_class
    return truth


def best_detection(result: Any) -> tuple[str, float, int]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "", 0.0, 0

    best_box = max(boxes, key=lambda box: float(box.conf[0]))
    class_id = int(best_box.cls[0])
    confidence = float(best_box.conf[0])
    return CLASS_NAMES.get(class_id, "unknown"), confidence, len(boxes)


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_one_policy(
    policy: str,
    weight: Path,
    source: Path,
    output_root: Path,
    conf: float,
    image_truth: dict[str, str],
) -> list[dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("\n" + "=" * 72)
    print(f"Testing policy: {policy}")
    print(f"Weight: {weight}")
    print("=" * 72)

    model = YOLO(str(weight))
    results = model.predict(
        source=str(source),
        conf=conf,
        save=True,
        project=str(output_root),
        name=policy,
        exist_ok=True,
        verbose=False,
    )

    rows: list[dict[str, Any]] = []
    for result in results:
        image_path = Path(result.path)
        pred_class, confidence, num_detections = best_detection(result)
        actual_class = image_truth.get(str(image_path), image_truth.get(image_path.name, ""))
        correct = ""
        if actual_class:
            correct = pred_class == actual_class

        rows.append(
            {
                "policy": policy,
                "image": image_path.name,
                "image_path": str(image_path),
                "actual_class": actual_class,
                "pred_class": pred_class,
                "confidence": round(confidence, 4),
                "num_detections": num_detections,
                "detected": bool(pred_class),
                "correct": correct,
                "output_folder": str(output_root / policy),
                "weight": str(weight),
            }
        )
    return rows


def summarize_policy(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    detected = sum(1 for row in rows if row["detected"])
    undetected = total - detected
    has_truth = any(row["actual_class"] for row in rows)
    correct_rows = [row for row in rows if row["correct"] is True]
    incorrect_rows = [row for row in rows if row["correct"] is False]

    summary: dict[str, Any] = {
        "policy": policy,
        "total_images": total,
        "detected": detected,
        "undetected": undetected,
        "ground_truth_available": has_truth,
        "correct": "",
        "incorrect": "",
        "accuracy": "",
        "closed_cap_predictions": sum(1 for row in rows if row["pred_class"] == "closed_cap"),
        "half_cap_predictions": sum(1 for row in rows if row["pred_class"] == "half_cap"),
        "no_cap_predictions": sum(1 for row in rows if row["pred_class"] == "no_cap"),
    }

    if has_truth:
        summary["correct"] = len(correct_rows)
        summary["incorrect"] = len(incorrect_rows)
        summary["accuracy"] = round(len(correct_rows) / total, 4) if total else ""

    return summary


def build_review_rows(images: list[Path], source: Path, long_rows: list[dict[str, Any]], truth_csv: dict[str, str]) -> list[dict[str, Any]]:
    by_image: dict[str, dict[str, Any]] = {}
    for image in images:
        actual = truth_csv.get(str(image), truth_csv.get(image.name, infer_truth_from_folder(image, source)))
        by_image[image.name] = {
            "image": image.name,
            "actual_class": actual,
            "manual_correct": "",
            "notes": "",
        }

    for row in long_rows:
        image_name = row["image"]
        policy = row["policy"]
        by_image.setdefault(image_name, {"image": image_name, "actual_class": "", "manual_correct": "", "notes": ""})
        by_image[image_name][f"{policy}_pred"] = row["pred_class"]
        by_image[image_name][f"{policy}_conf"] = row["confidence"]

    return [by_image[image.name] for image in images]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test all YOLO augmentation weights on one image set.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help=f"Image file/folder (default: {DEFAULT_SOURCE})")
    parser.add_argument("--weights-root", default=str(AUG_ROOT), help=f"Augmentation experiment folder (default: {AUG_ROOT})")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help=f"Output root folder (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--truth-csv", default="", help="Optional CSV with image,actual_class columns")
    parser.add_argument("--run-name", default="", help="Optional output run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    weights_root = Path(args.weights_root).resolve()
    output_root = Path(args.output_dir).resolve()
    run_name = args.run_name or f"aug_weight_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source)
    truth_csv = load_truth_csv(Path(args.truth_csv).resolve() if args.truth_csv else None)
    image_truth = {
        str(image): truth_csv.get(str(image), truth_csv.get(image.name, infer_truth_from_folder(image, source)))
        for image in images
    }

    weights = find_weights(weights_root)
    print("\nYOLO augmentation weight test")
    print(f"Source       : {source}")
    print(f"Images       : {len(images)}")
    print(f"Weights root : {weights_root}")
    print(f"Output       : {run_dir}")
    print(f"Confidence   : {args.conf}")
    print(f"Ground truth : {'yes' if any(image_truth.values()) else 'no'}")

    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for policy, weight in weights.items():
        rows = test_one_policy(policy, weight, source, run_dir, args.conf, image_truth)
        long_rows.extend(rows)
        summary_rows.append(summarize_policy(policy, rows))

    long_headers = [
        "policy",
        "image",
        "image_path",
        "actual_class",
        "pred_class",
        "confidence",
        "num_detections",
        "detected",
        "correct",
        "output_folder",
        "weight",
    ]
    summary_headers = [
        "policy",
        "total_images",
        "detected",
        "undetected",
        "ground_truth_available",
        "correct",
        "incorrect",
        "accuracy",
        "closed_cap_predictions",
        "half_cap_predictions",
        "no_cap_predictions",
    ]
    review_headers = ["image", "actual_class", "manual_correct", "notes"]
    for policy in weights:
        review_headers.extend([f"{policy}_pred", f"{policy}_conf"])

    review_rows = build_review_rows(images, source, long_rows, truth_csv)

    write_csv(run_dir / "predictions_long.csv", long_rows, long_headers)
    write_csv(run_dir / "summary.csv", summary_rows, summary_headers)
    write_csv(run_dir / "predictions_review.csv", review_rows, review_headers)

    print("\nSummary")
    print("-" * 72)
    for row in summary_rows:
        if row["ground_truth_available"]:
            print(
                f"{row['policy']:<20} detected={row['detected']}/{row['total_images']} "
                f"correct={row['correct']}/{row['total_images']} accuracy={row['accuracy']}"
            )
        else:
            print(
                f"{row['policy']:<20} detected={row['detected']}/{row['total_images']} "
                "correct=N/A (no ground truth)"
            )

    print("\nFiles written:")
    print(f"  {run_dir / 'summary.csv'}")
    print(f"  {run_dir / 'predictions_long.csv'}")
    print(f"  {run_dir / 'predictions_review.csv'}")
    print("\nPrediction image folders:")
    for policy in weights:
        print(f"  {run_dir / policy}")

    if not any(image_truth.values()):
        print("\nNote: Correct/incorrect cannot be computed automatically because this source has no ground truth.")
        print("Fill actual_class/manual_correct in predictions_review.csv, or use a source folder with class subfolders.")


if __name__ == "__main__":
    main()
