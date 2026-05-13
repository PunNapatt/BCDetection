#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   Auto-Labeling Script — Bottle Cap Detection                   ║
║   Model  : Grounding DINO Tiny (IDEA-Research / HuggingFace)    ║
║   Output : YOLO format .txt labels  (class x_c y_c w h)         ║
╠══════════════════════════════════════════════════════════════════╣
║  Classes                                                         ║
║    0 : closed_cap  —  ฝาปิดสนิท                                  ║
║    1 : half_cap    —  ฝาปิดไม่สนิท                               ║
║    2 : no_cap      —  ไม่มีฝา                                    ║
╚══════════════════════════════════════════════════════════════════╝

Usage
-----
  # Basic (auto-detect GPU/CPU)
  python auto_label.py

  # Force CPU
  python auto_label.py --device cpu

  # Try first without writing files
  python auto_label.py --dry-run

  # Adjust detection confidence (default varies per class)
  python auto_label.py --threshold 0.30

  # Visualise boxes after labeling (saves *_preview.jpg)
  python auto_label.py --visualise
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset root  (script lives inside  bottle_cap - ver 2/)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
#  Class configuration
#  prompt  : text sent to Grounding DINO  (dot-separated phrases work best)
#  threshold: minimum confidence to keep a box
# ─────────────────────────────────────────────────────────────────────────────
CLASS_CONFIG: dict[str, dict] = {
    "closed_cap": {
        "class_id":  0,
        "prompt":    "bottle cap . closed bottle cap . sealed cap .",
        "threshold": 0.30,
    },
    "half_cap": {
        "class_id":  1,
        "prompt":    "bottle cap . partially open cap . loose cap . half open bottle cap .",
        "threshold": 0.28,
    },
    "no_cap": {
        "class_id":  2,
        "prompt":    "bottle . open bottle . bottle neck . bottle without cap .",
        "threshold": 0.28,
    },
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Colour per class for visualisation
VIS_COLORS = {0: (0, 200, 0), 1: (255, 165, 0), 2: (220, 50, 50)}


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(device: str):
    """Load Grounding DINO Tiny from HuggingFace Transformers."""
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        print("[ERROR] transformers not found.  Run:  pip install transformers")
        sys.exit(1)

    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"  Downloading / loading '{model_id}' …")

    processor = AutoProcessor.from_pretrained(model_id)
    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(model_id)
        .to(device)
    )
    model.eval()
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict(
    processor,
    model,
    image: Image.Image,
    text_prompt: str,
    threshold: float,
    device: str,
) -> list[tuple[float, float, float, float, float]]:
    """
    Run Grounding DINO on one image.

    Returns
    -------
    list of (x_center, y_center, width, height, score)
    All coordinates are normalised to [0, 1].

    Note: post-processes outputs manually to stay compatible across
    all transformers versions (the processor API changed in >=4.44).
    """
    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # ── Manual post-processing (version-agnostic) ──────────────────────────
    # outputs.logits  : (1, num_queries, num_text_tokens)  — already sigmoid-ready
    # outputs.pred_boxes: (1, num_queries, 4)              — cxcywh, normalised [0,1]
    logits = outputs.logits[0].sigmoid()       # (num_queries, num_text_tokens)
    pred_boxes = outputs.pred_boxes[0]         # (num_queries, 4)  cx cy w h  in [0,1]

    # Score per box = max similarity over all text tokens
    scores = logits.max(dim=-1).values         # (num_queries,)

    # Filter by threshold
    keep = scores > threshold
    kept_scores = scores[keep]
    kept_boxes  = pred_boxes[keep]             # still in cxcywh normalised — perfect for YOLO

    yolo_boxes: list[tuple[float, float, float, float, float]] = []
    for box, score in zip(kept_boxes, kept_scores):
        xc, yc, bw, bh = box.tolist()
        # clamp to [0, 1]
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        yolo_boxes.append((xc, yc, bw, bh, float(score)))

    return yolo_boxes


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────
def draw_boxes(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float, float]],
    class_id: int,
    class_name: str,
) -> Image.Image:
    """Return a copy of *image* with bounding boxes drawn."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    color = VIS_COLORS.get(class_id, (255, 255, 0))

    for xc, yc, bw, bh, score in boxes:
        x1 = int((xc - bw / 2) * W)
        y1 = int((yc - bh / 2) * H)
        x2 = int((xc + bw / 2) * W)
        y2 = int((yc + bh / 2) * H)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{class_name} {score:.2f}"
        draw.rectangle([x1, y1 - 18, x1 + len(label) * 8, y1], fill=color)
        draw.text((x1 + 2, y1 - 17), label, fill=(255, 255, 255))

    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Per-class processing
# ─────────────────────────────────────────────────────────────────────────────
def process_class(
    class_name: str,
    cfg: dict,
    processor,
    model,
    device: str,
    dry_run: bool,
    visualise: bool,
) -> dict:
    folder = DATASET_DIR / class_name
    if not folder.exists():
        print(f"  [SKIP] folder not found: {folder}")
        return {"found": 0, "labeled": 0, "no_detect": 0, "skipped": 0}

    images = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    stats = {"found": len(images), "labeled": 0, "no_detect": 0, "skipped": 0}

    if not images:
        print(f"  [SKIP] no images in {folder}")
        return stats

    for img_path in tqdm(images, desc=f"  {class_name}", unit="img", leave=False):
        # ── load image ──────────────────────────────────────────────────────
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            tqdm.write(f"    [ERROR] {img_path.name}: {exc}")
            stats["skipped"] += 1
            continue

        # ── inference ───────────────────────────────────────────────────────
        boxes = predict(
            processor, model, image,
            cfg["prompt"], cfg["threshold"], device,
        )

        # ── fallback: full-frame box when nothing detected ───────────────────
        if not boxes:
            boxes = [(0.5, 0.5, 1.0, 1.0, 0.0)]
            stats["no_detect"] += 1
            tqdm.write(
                f"    [WARN] no detection → full-frame fallback: {img_path.name}"
            )

        # ── write YOLO label ─────────────────────────────────────────────────
        if not dry_run:
            label_path = img_path.with_suffix(".txt")
            with open(label_path, "w") as fh:
                for xc, yc, bw, bh, _ in boxes:
                    fh.write(
                        f"{cfg['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
                    )

            # ── optional visualisation ───────────────────────────────────────
            if visualise:
                vis_img = draw_boxes(image, boxes, cfg["class_id"], class_name)
                vis_path = img_path.with_name(img_path.stem + "_preview.jpg")
                vis_img.save(vis_path, quality=85)

        stats["labeled"] += 1

    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  dataset.yaml
# ─────────────────────────────────────────────────────────────────────────────
def write_yaml():
    yaml_path = DATASET_DIR / "dataset.yaml"
    content = (
        "# YOLO dataset config — generated by auto_label.py\n"
        f"path: {DATASET_DIR.resolve()}\n\n"
        "# Update these after running split_dataset.py\n"
        "train: images/train\n"
        "val:   images/val\n\n"
        "nc: 3\n"
        "names:\n"
        "  0: closed_cap\n"
        "  1: half_cap\n"
        "  2: no_cap\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"  dataset.yaml  →  {yaml_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-label bottle cap images with Grounding DINO",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: cuda | cpu | auto  (default: auto)",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Override confidence threshold for ALL classes (e.g. 0.30)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Run inference but do NOT write any .txt or .yaml files",
    )
    p.add_argument(
        "--visualise", action="store_true",
        help="Save *_preview.jpg alongside each label to inspect detections",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── device ──────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\n{'='*60}")
    print(f"  Bottle Cap Auto-Labeler  (Grounding DINO)")
    print(f"{'='*60}")
    print(f"  Device      : {device.upper()}")
    print(f"  Dataset dir : {DATASET_DIR}")
    print(f"  Dry-run     : {args.dry_run}")
    print(f"  Visualise   : {args.visualise}")

    # ── threshold override ───────────────────────────────────────────────────
    if args.threshold is not None:
        for cfg in CLASS_CONFIG.values():
            cfg["threshold"] = args.threshold
        print(f"  Threshold   : {args.threshold} (all classes)")
    print()

    # ── load model ───────────────────────────────────────────────────────────
    print("Loading model …")
    processor, model = load_model(device)
    print("  Model ready!\n")

    # ── process each class ───────────────────────────────────────────────────
    totals = {"found": 0, "labeled": 0, "no_detect": 0, "skipped": 0}

    for class_name, cfg in CLASS_CONFIG.items():
        print(f"[{class_name}]")
        print(f"  prompt     : {cfg['prompt']}")
        print(f"  threshold  : {cfg['threshold']}")
        stats = process_class(
            class_name, cfg, processor, model, device,
            dry_run=args.dry_run,
            visualise=args.visualise,
        )
        print(
            f"  found={stats['found']}  labeled={stats['labeled']}"
            f"  no_detect={stats['no_detect']}  skipped={stats['skipped']}\n"
        )
        for k in totals:
            totals[k] += stats[k]

    # summary
    n_found   = totals['found']
    n_labeled = totals['labeled']
    n_fallbk  = totals['no_detect']
    n_skip    = totals['skipped']
    print('=' * 60)
    print(
        '  DONE  |  total=' + str(n_found) +
        '  labeled=' + str(n_labeled) +
        '  fallback=' + str(n_fallbk) +
        '  skipped=' + str(n_skip)
    )
    if args.dry_run:
        print('  (dry-run -- no files written)')
    else:
        print('\nWriting dataset.yaml ...')
        write_yaml()
        print(
            '\nNext step: run  split_dataset.py  to create train/val splits'
            '\nthen train: yolo train data=dataset.yaml model=yolov8n.pt epochs=100'
        )
    print('=' * 60)


if __name__ == '__main__':
    main()
