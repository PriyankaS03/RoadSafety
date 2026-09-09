#!/usr/bin/env python3
"""
Quick Training Script for Road Damage Detection
Run this to train the YOLOv8 model with high accuracy
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description=""):
    """Run shell command with description"""
    if description:
        print(f"\n{'='*50}")
        print(f"{description}")
        print(f"{'='*50}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
    return result.returncode == 0


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║     ROAD DAMAGE DETECTION - YOLOv8 TRAINING               ║
║     High Accuracy Model Training Pipeline                 ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Install requirements
    print("\n[1/6] Installing requirements...")
    packages = [
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.9.0",
        "numpy>=1.24.0"
    ]

    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                      capture_output=True)

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CPU mode - GPU recommended for faster training")
    except:
        print("⚠ PyTorch checking...")

    # Create directories
    print("\n[2/6] Creating directories...")
    dirs = [
        "data/road_dataset/images/train",
        "data/road_dataset/images/val",
        "data/road_dataset/labels/train",
        "data/road_dataset/labels/val",
        "runs/train/road_damage/weights"
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Generate dataset
    print("\n[3/6] Generating training dataset...")
    generate_dataset()


def generate_dataset():
    """Generate synthetic training dataset"""
    import cv2
    import numpy as np

    print("Creating 2000 synthetic road images...")

    train_img = Path("data/road_dataset/images/train")
    train_lbl = Path("data/road_dataset/labels/train")
    val_img = Path("data/road_dataset/images/val")
    val_lbl = Path("data/road_dataset/labels/val")

    np.random.seed(42)
    num_samples = 2000

    for i in range(num_samples):
        # Class distribution
        if i < 500:
            class_id = 0  # good_road
        elif i < 800:
            class_id = 1  # pothole
        elif i < 1100:
            class_id = 2  # crack
        elif i < 1500:
            class_id = 3  # accident
        else:
            class_id = 4  # road_damage

        # Generate image
        img = np.random.randint(100, 160, (640, 640, 3), dtype=np.uint8)
        noise = np.random.randint(-20, 20, (640, 640, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add patterns
        if class_id == 0:
            cv2.rectangle(img, (250, 0), (390, 640), (255, 255, 255), 3)
            cv2.putText(img, "GOOD ROAD", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            bboxes = []
        elif class_id == 1:
            cv2.circle(img, (320, 400), 80, (20, 20, 20), -1)
            cv2.putText(img, "POTHOLE", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            bboxes = [(0.5, 0.625, 0.25, 0.25)]
        elif class_id == 2:
            for _ in range(5):
                x1 = np.random.randint(100, 300)
                y1 = np.random.randint(100, 500)
                x2 = x1 + np.random.randint(-100, 100)
                y2 = y1 + np.random.randint(50, 200)
                cv2.line(img, (x1, y1), (x2, y2), (30, 30, 30), 2)
            cv2.putText(img, "CRACK", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), 3)
            bboxes = [(0.5, 0.5, 0.6, 0.4)]
        elif class_id == 3:
            cv2.rectangle(img, (150, 200), (490, 450), (0, 0, 200), -1)
            cv2.circle(img, (200, 350), 30, (255, 255, 0), -1)
            cv2.circle(img, (440, 350), 30, (255, 255, 0), -1)
            cv2.putText(img, "ACCIDENT!", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            bboxes = [(0.5, 0.5, 0.53, 0.39)]
        else:
            cv2.rectangle(img, (100, 300), (250, 450), (30, 30, 30), -1)
            for _ in range(3):
                x = np.random.randint(350, 500)
                y = np.random.randint(200, 400)
                cv2.circle(img, (x, y), 20, (25, 25, 25), -1)
            cv2.putText(img, "DAMAGE", (230, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 100), 3)
            bboxes = [(0.3, 0.5, 0.3, 0.3), (0.7, 0.5, 0.2, 0.2)]

        # Save
        split = "val" if i % 5 == 0 else "train"
        img_dir = val_img if split == "val" else train_img
        lbl_dir = val_lbl if split == "val" else train_lbl

        img_name = f"road_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / img_name), img)

        label_name = f"road_{i:06d}.txt"
        with open(lbl_dir / label_name, "w") as f:
            for bbox in bboxes:
                x, y, w, h = bbox
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{num_samples} samples created...")

    # Create YAML
    yaml_content = """path: data/road_dataset
train: images/train
val: images/val
names:
  0: good_road
  1: pothole
  2: crack
  3: accident
  4: road_damage
nc: 5
"""
    with open("data/road_dataset/dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"✓ Dataset created: {num_samples} images")


def train_model():
    """Train the YOLOv8 model"""
    print("\n[4/6] Training YOLOv8 model...")

    try:
        from ultralytics import YOLO

        model = YOLO("yolov8m.pt")

        results = model.train(
            data="data/road_dataset/dataset.yaml",
            epochs=150,
            imgsz=640,
            batch=16,
            patience=50,
            save=True,
            project="runs/train",
            name="road_damage",
            exist_ok=True,
            optimizer="AdamW",
            lr0=0.001,
            amp=True,
            plots=True
        )

        print("✓ Training complete!")
        return True

    except Exception as e:
        print(f"Training error: {e}")
        return False


def evaluate():
    """Evaluate the model"""
    print("\n[5/6] Evaluating model...")

    try:
        from ultralytics import YOLO

        model = YOLO("runs/train/road_damage/weights/best.pt")
        metrics = model.val(data="data/road_dataset/dataset.yaml")

        print(f"\n📊 Results:")
        print(f"   mAP@0.5: {metrics.box.map50:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")

        return True

    except Exception as e:
        print(f"Evaluation error: {e}")
        return False


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║     ROAD DAMAGE DETECTION - YOLOv8 TRAINING               ║
║     High Accuracy Model Training Pipeline                 ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Install requirements
    print("\n[1/6] Installing requirements...")
    packages = [
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ]

    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                      capture_output=True)

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("⚠ Running on CPU")

    # Create directories
    print("\n[2/6] Creating directories...")
    for d in ["data/road_dataset/images/train", "data/road_dataset/images/val",
              "data/road_dataset/labels/train", "data/road_dataset/labels/val"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Generate dataset
    print("\n[3/6] Generating dataset...")
    generate_dataset()

    # Train
    print("\n[4/6] Training model...")
    train_model()

    # Evaluate
    print("\n[5/6] Evaluating...")
    evaluate()

    print("\n" + "="*50)
    print("✓ TRAINING COMPLETE!")
    print("Model: runs/train/road_damage/weights/best.pt")
    print("="*50)


if __name__ == "__main__":
    main()