#!/usr/bin/env python3
"""
Bottle Cap Detection — Inference Script
========================================
Uses trained YOLOv8 best.pt to detect:
  0 = closed_cap  (ฝาปิดสนิท)
  1 = half_cap    (ฝาปิดไม่สนิท)
  2 = no_cap      (ไม่มีฝา)

Usage
-----
  # ทดสอบภาพเดียว
  python predict.py --source photo.jpg

  # ทดสอบทั้งโฟลเดอร์
  python predict.py --source ./test_images/

  # กล้อง webcam แบบ real-time
  python predict.py --source webcam

  # กำหนด weights เอง
  python predict.py --source photo.jpg --weights path/to/best.pt

  # ปรับ confidence threshold
  python predict.py --source photo.jpg --conf 0.5
"""

import argparse
import sys
from pathlib import Path

# Default to the best augmentation experiment weight (No Mosaic).
DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent
    / "yolo_augmentation_output"
    / "aug_compare_20260512_181523"
    / "no_mosaic"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = {0: "closed_cap", 1: "half_cap", 2: "no_cap"}
CLASS_TH    = {
    "closed_cap": "ฝาปิดสนิท   ✅",
    "half_cap":   "ฝาปิดไม่สนิท ⚠️",
    "no_cap":     "ไม่มีฝา      ❌",
}


def load_model(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.  pip install ultralytics")
        sys.exit(1)
    if not weights.exists():
        print(f"[ERROR] weights file not found: {weights}")
        sys.exit(1)
    print(f"Loading model: {weights}")
    return YOLO(str(weights))


# ─────────────────────────────────────────────────────────────────────────────
#  1. Predict ภาพเดียว / โฟลเดอร์  — บันทึกผลเป็นรูป
# ─────────────────────────────────────────────────────────────────────────────
def predict_images(model, source: str, conf: float, save_dir: str) -> None:
    results = model.predict(
        source   = source,
        conf     = conf,
        save     = True,
        project  = save_dir,
        name     = "output",
        exist_ok = True,
    )
    print(f"\nResults saved to: {save_dir}/output/")
    print("-" * 50)
    for r in results:
        img_name = Path(r.path).name
        boxes    = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  {img_name}: ไม่พบวัตถุ")
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_s = float(box.conf[0])
            label  = CLASS_NAMES.get(cls_id, "unknown")
            thai   = CLASS_TH.get(label, label)
            print(f"  {img_name}: {thai}  (conf={conf_s:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
#  2. Real-time webcam
# ─────────────────────────────────────────────────────────────────────────────
def predict_webcam(model, conf: float, cam_id: int = 0) -> None:
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python not installed.  pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}")
        sys.exit(1)

    print("Real-time detection  (กด Q เพื่อออก)")
    COLOR = {0: (0, 200, 0), 1: (0, 165, 255), 2: (50, 50, 220)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        r = results[0]

        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id  = int(box.cls[0])
                conf_s  = float(box.conf[0])
                label   = CLASS_NAMES.get(cls_id, "?")
                color   = COLOR.get(cls_id, (255, 255, 0))
                text    = f"{label} {conf_s:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(text) * 11, y1), color, -1)
                cv2.putText(frame, text, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Bottle Cap Detection  (Q=quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  3. Import ใช้เป็น function ใน code อื่น
# ─────────────────────────────────────────────────────────────────────────────
def detect(image_path: str, weights: Path = DEFAULT_WEIGHTS, conf: float = 0.4) -> list[dict]:
    """
    ใช้เรียกจาก code อื่น

    Returns
    -------
    list of dict  e.g.  [{"class": "closed_cap", "conf": 0.97, "box": [x1,y1,x2,y2]}, ...]
    """
    model = load_model(weights)
    results = model.predict(image_path, conf=conf, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": CLASS_NAMES.get(cls_id, "unknown"),
                "conf":  round(float(box.conf[0]), 4),
                "box":   [round(v, 1) for v in box.xyxy[0].tolist()],
            })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bottle Cap Detection — Inference")
    p.add_argument("--source",  required=True,
                   help="image path / folder / 'webcam'")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                   help=f"path to best.pt  (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--conf",    type=float, default=0.4,
                   help="confidence threshold  (default: 0.4)")
    p.add_argument("--save-dir", default="./predictions",
                   help="output folder for saved images  (default: ./predictions)")
    p.add_argument("--cam-id",  type=int, default=0,
                   help="webcam device id  (default: 0)")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    model = load_model(Path(args.weights))

    if args.source.lower() == "webcam":
        predict_webcam(model, conf=args.conf, cam_id=args.cam_id)
    else:
        predict_images(model, source=args.source, conf=args.conf, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
