"""
advanced_train.py – Advanced Road Damage Model Training Pipeline
================================================================
Trains a YOLOv8x model (or larger) for accurate road damage detection.

Classes:
  0: pothole       – Potholes and cavities
  1: crack         – Surface cracks (longitudinal / transverse / alligator)
  2: edge_break    – Road edge deterioration
  3: rutting       – Wheel-path deformation / ruts
  4: road_damage   – General surface damage

Features:
  • High-quality synthetic dataset with realistic road textures
  • Correct YOLO-format labels (normalized xywh)
  • Advanced augmentation: mosaic, mixup, copy-paste, HSV jitter
  • Cosine-LR schedule with warm-up
  • Early stopping with patience
  • Automatic best-weights export

Usage:
  python advanced_train.py --epochs 150 --model large --synthetic
  python advanced_train.py --data path/to/real/dataset --epochs 200
"""

import os
import sys
import yaml
import math
import random
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ── Ultralytics ────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    import torch
except ImportError:
    os.system("pip install ultralytics torch torchvision --quiet")
    from ultralytics import YOLO
    import torch


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "pothole",
    1: "crack",
    2: "edge_break",
    3: "rutting",
    4: "road_damage",
}
NC = len(CLASS_NAMES)
IMG_SIZE = 640


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Image Generators
# ─────────────────────────────────────────────────────────────────────────────

def _base_road(size: int = 640) -> np.ndarray:
    """Generate a realistic asphalt-like base texture."""
    # Randomise base grey value (wet/dry road)
    base = random.randint(70, 145)
    img  = np.full((size, size, 3), base, dtype=np.uint8)

    # Perlin-like noise via overlapping Gaussian blur layers
    noise = np.random.randint(-18, 18, (size, size, 3), dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Coarse noise (aggregate texture)
    coarse = np.random.randint(-10, 10, (size // 8, size // 8, 3), dtype=np.int16)
    coarse = cv2.resize(coarse.astype(np.uint8), (size, size), interpolation=cv2.INTER_CUBIC)
    img    = np.clip(img.astype(np.int16) + coarse.astype(np.int16) - 5, 0, 255).astype(np.uint8)

    # Random brightness gradient (lighting)
    grad = np.linspace(random.uniform(0.85, 1.0), random.uniform(0.88, 1.0), size)
    grad = np.outer(grad, np.ones(size)).astype(np.float32)
    img  = np.clip((img.astype(np.float32) * grad[:, :, None]), 0, 255).astype(np.uint8)

    # Optional lane marking
    if random.random() < 0.5:
        lw = random.randint(3, 6)
        cx = random.randint(size // 3, 2 * size // 3)
        cv2.line(img, (cx, 0), (cx, size), (220, 220, 220), lw)

    # Optional road edge marking
    if random.random() < 0.3:
        ew = random.randint(2, 4)
        cv2.line(img, (12, 0), (12, size), (200, 190, 160), ew)
        cv2.line(img, (size - 12, 0), (size - 12, size), (200, 190, 160), ew)

    return img


def _add_pothole(img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    h, w = img.shape[:2]
    # Random position (lower 60% — on the road)
    cx = random.randint(w // 5, 4 * w // 5)
    cy = random.randint(int(h * 0.40), int(h * 0.90))
    rx = random.randint(18, 70)
    ry = random.randint(12, 55)

    # Dark fill with rough edges
    dark = max(10, img[cy, cx, 0].item() - random.randint(40, 80))
    cv2.ellipse(img, (cx, cy), (rx, ry), random.uniform(0, 180), 0, 360, (dark, dark, dark - 5), -1)

    # Rough border
    for angle in range(0, 360, random.randint(20, 40)):
        rad = math.radians(angle)
        ex  = int(cx + (rx + random.randint(2, 10)) * math.cos(rad))
        ey  = int(cy + (ry + random.randint(2, 8)) * math.sin(rad))
        cv2.circle(img, (ex, ey), random.randint(3, 8), (dark + 8, dark + 8, dark + 3), -1)

    # Rim highlight
    cv2.ellipse(img, (cx, cy), (rx + 3, ry + 3), 0, 0, 360, (min(255, dark + 30),) * 3, 2)

    # Normalised YOLO bbox
    x1, y1 = cx - rx - 8, cy - ry - 8
    x2, y2 = cx + rx + 8, cy + ry + 8
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return img, [0, xc, yc, bw, bh]   # class 0 = pothole


def _add_crack(img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    h, w = img.shape[:2]
    x1 = random.randint(w // 6, w // 2)
    y1 = random.randint(int(h * 0.30), int(h * 0.85))
    crack_clr = max(15, img[y1, x1, 0].item() - random.randint(30, 65))
    pts = [(x1, y1)]
    for _ in range(random.randint(4, 9)):
        dx = random.randint(-50, 80)
        dy = random.randint(15, 60)
        nx, ny = pts[-1][0] + dx, pts[-1][1] + dy
        cv2.line(img, pts[-1], (nx, ny), (crack_clr,) * 3, random.randint(1, 3))
        pts.append((nx, ny))
        # Branch crack
        if random.random() < 0.35:
            bx = nx + random.randint(-35, 35)
            by = ny + random.randint(10, 40)
            cv2.line(img, (nx, ny), (bx, by), (crack_clr + 4,) * 3, 1)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1b, y1b = max(0, min(xs) - 6), max(0, min(ys) - 6)
    x2b, y2b = min(w, max(xs) + 6), min(h, max(ys) + 6)
    xc = (x1b + x2b) / 2 / w
    yc = (y1b + y2b) / 2 / h
    bw = (x2b - x1b) / w
    bh = (y2b - y1b) / h
    return img, [1, xc, yc, max(0.02, bw), max(0.02, bh)]   # class 1 = crack


def _add_edge_break(img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    h, w = img.shape[:2]
    side    = random.choice(['left', 'right'])
    ex      = 0 if side == 'left' else w - 1
    ey      = random.randint(int(h * 0.3), int(h * 0.9))
    bw_px   = random.randint(30, 90)
    bh_px   = random.randint(40, 120)
    dark    = max(20, img[ey, min(ex, w - 1), 0].item() - 40)

    pts = np.array([
        [ex, ey],
        [ex + (bw_px if side == 'left' else -bw_px), ey],
        [ex + (bw_px // 2 if side == 'left' else -bw_px // 2), ey + bh_px],
        [ex, ey + bh_px],
    ], np.int32)
    cv2.fillPoly(img, [pts], (dark, dark - 5, dark - 10))

    xs, ys = pts[:, 0], pts[:, 1]
    x1, y1 = max(0, xs.min() - 5), max(0, ys.min() - 5)
    x2, y2 = min(w, xs.max() + 5), min(h, ys.max() + 5)
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    return img, [2, xc, yc, (x2 - x1) / w, (y2 - y1) / h]  # class 2


def _add_rutting(img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    h, w = img.shape[:2]
    track_x = random.randint(w // 5, 4 * w // 5)
    y_start  = random.randint(int(h * 0.30), int(h * 0.55))
    rut_len  = random.randint(80, h - y_start - 20)
    rut_w    = random.randint(12, 30)

    dark = max(20, int(img[y_start, track_x, 0]) - random.randint(20, 45))
    for y in range(y_start, y_start + rut_len):
        jitter = random.randint(-3, 3)
        cv2.line(img, (track_x - rut_w // 2 + jitter, y),
                 (track_x + rut_w // 2 + jitter, y),
                 (dark, dark, dark - 5), 1)

    x1 = max(0, track_x - rut_w // 2 - 5)
    x2 = min(w, track_x + rut_w // 2 + 5)
    y1, y2 = y_start, y_start + rut_len
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    return img, [3, xc, yc, (x2 - x1) / w, (y2 - y1) / h]  # class 3


def _add_road_damage(img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """Alligator / combined damage patch."""
    h, w = img.shape[:2]
    cx  = random.randint(w // 5, 4 * w // 5)
    cy  = random.randint(int(h * 0.40), int(h * 0.85))
    dpw = random.randint(50, 140)
    dph = random.randint(40, 110)

    dark = max(20, int(img[cy, cx, 0]) - random.randint(25, 55))
    cv2.rectangle(img, (cx - dpw // 2, cy - dph // 2),
                  (cx + dpw // 2, cy + dph // 2), (dark,) * 3, -1)

    # Alligator crack lines
    for _ in range(random.randint(4, 12)):
        ax = random.randint(cx - dpw // 2, cx + dpw // 2)
        ay = random.randint(cy - dph // 2, cy + dph // 2)
        bx = ax + random.randint(-30, 30)
        by = ay + random.randint(-25, 25)
        cv2.line(img, (ax, ay), (bx, by), (dark - 10,) * 3, 1)

    x1 = max(0, cx - dpw // 2 - 5)
    x2 = min(w, cx + dpw // 2 + 5)
    y1 = max(0, cy - dph // 2 - 5)
    y2 = min(h, cy + dph // 2 + 5)
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    return img, [4, xc, yc, (x2 - x1) / w, (y2 - y1) / h]  # class 4


def _generate_good_road(size: int = 640) -> Tuple[np.ndarray, List]:
    """Clean road — no labels."""
    return _base_road(size), []


def _generate_sample(class_id: int, size: int = 640) -> Tuple[np.ndarray, List]:
    img = _base_road(size)
    generators = {
        0: _add_pothole,
        1: _add_crack,
        2: _add_edge_break,
        3: _add_rutting,
        4: _add_road_damage,
    }
    gen = generators.get(class_id)
    if gen is None:
        return img, []
    img, label = gen(img)

    # Optionally add a second instance for multi-object training
    if random.random() < 0.30 and class_id != 2:
        try:
            img, label2 = gen(img)
            return img, [label, label2]
        except Exception:
            pass
    return img, [label]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset creation
# ─────────────────────────────────────────────────────────────────────────────

def create_synthetic_dataset(
    output_dir: str = "data/road_dataset_adv",
    num_samples: int = 3000,
    val_frac: float = 0.15,
) -> str:
    out = Path(output_dir)
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} synthetic road images → {out}")

    # Class distribution: 25% good, rest damage classes equally
    n_good   = int(num_samples * 0.25)
    n_damage = num_samples - n_good
    per_cls  = n_damage // NC

    samples: List[Tuple[int, bool]] = (
        [(-1, False)] * n_good
        + [(c, False) for c in range(NC) for _ in range(per_cls)]
    )
    random.shuffle(samples)

    for idx, (cls, _) in enumerate(samples):
        if cls == -1:
            img, labels = _generate_good_road()
        else:
            img, labels = _generate_sample(cls)

        split  = "val" if random.random() < val_frac else "train"
        name   = f"road_{idx:06d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        lbl_path = out / "labels" / split / f"{name}.txt"
        with open(lbl_path, "w") as f:
            if isinstance(labels, list) and labels:
                for row in labels:
                    if isinstance(row, list) and len(row) == 5:
                        f.write(" ".join(f"{v:.6f}" if i > 0 else str(int(v))
                                         for i, v in enumerate(row)) + "\n")

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{num_samples} generated …")

    print(f"✅ Dataset ready: {out}")
    return str(out)


def create_dataset_yaml(dataset_path: str) -> str:
    data = {
        "path":  str(Path(dataset_path).resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NC,
        "names": {i: n for i, n in CLASS_NAMES.items()},
    }
    yaml_path = Path(dataset_path) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"📄 YAML config: {yaml_path}")
    return str(yaml_path)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

MODEL_MAP = {
    "nano":   "yolov8n.pt",
    "small":  "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large":  "yolov8l.pt",
    "xlarge": "yolov8x.pt",
}


def train(
    data_yaml: str,
    model_size: str = "medium",
    epochs: int = 150,
    batch: int = 16,
    patience: int = 40,
    output_dir: str = "runs/train",
    resume: bool = False,
) -> str:
    use_gpu = torch.cuda.is_available()
    device  = "0" if use_gpu else "cpu"
    print(f"\n{'='*60}")
    print(f"  VigiRoad Advanced Training — YOLOv8-{model_size.upper()}")
    print(f"  Device : {'GPU (' + torch.cuda.get_device_name(0) + ')' if use_gpu else 'CPU'}")
    print(f"  Epochs : {epochs}  |  Batch : {batch}  |  Patience : {patience}")
    print(f"{'='*60}\n")

    model_name = MODEL_MAP.get(model_size, "yolov8m.pt")
    model      = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=batch,
        patience=patience,
        device=device,

        # Output
        project=output_dir,
        name="road_damage",
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,

        # Optimiser — AdamW with cosine LR
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.005,
        momentum=0.937,
        weight_decay=0.0005,

        # Warm-up
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss weights
        box=7.5,
        cls=1.5,
        dfl=1.5,

        # Advanced augmentation
        mosaic=1.0,
        mixup=0.20,
        copy_paste=0.15,
        degrees=8.0,
        translate=0.15,
        scale=0.6,
        shear=3.0,
        perspective=0.0002,
        flipud=0.0,
        fliplr=0.50,
        hsv_h=0.020,
        hsv_s=0.70,
        hsv_v=0.40,

        # Misc
        amp=True,
        close_mosaic=15,
        verbose=True,
        resume=resume,
    )

    best = Path(output_dir) / "road_damage" / "weights" / "best.pt"
    if not best.exists():
        best = Path(output_dir) / "road_damage" / "weights" / "last.pt"

    print(f"\n✅ Training complete! Best weights: {best}")
    return str(best)


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────

def validate(model_path: str, data_yaml: str):
    model   = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=IMG_SIZE)
    print(f"\n📊 Validation Results")
    print(f"   mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   Precision   : {metrics.box.mp:.4f}")
    print(f"   Recall      : {metrics.box.mr:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VigiRoad Advanced Training")
    parser.add_argument("--model",     default="medium",
                        choices=list(MODEL_MAP.keys()),
                        help="YOLOv8 model size")
    parser.add_argument("--epochs",    type=int,  default=150)
    parser.add_argument("--batch",     type=int,  default=16)
    parser.add_argument("--patience",  type=int,  default=40)
    parser.add_argument("--data",      default="data/road_dataset_adv",
                        help="Dataset directory (created if --synthetic)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic dataset before training")
    parser.add_argument("--samples",   type=int,  default=3000,
                        help="Number of synthetic samples")
    parser.add_argument("--validate",  action="store_true",
                        help="Run validation after training")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume previous training run")
    args = parser.parse_args()

    data_path = args.data

    if args.synthetic:
        data_path = create_synthetic_dataset(
            output_dir=args.data,
            num_samples=args.samples,
        )

    yaml_path = create_dataset_yaml(data_path)

    best_model = train(
        data_yaml=yaml_path,
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        resume=args.resume,
    )

    if args.validate:
        validate(best_model, yaml_path)

    print(f"\n🚀 Model ready. Copy to: runs/train/road_damage/weights/best.pt")
    print(f"   Then restart the VigiRoad app to use the new model.")
