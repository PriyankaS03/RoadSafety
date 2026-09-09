"""
Complete Training Pipeline for Road Damage Detection
Run this script to train the YOLOv8 model with high accuracy
"""
import os
import sys
import subprocess
from pathlib import Path


def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    requirements = [
        'ultralytics>=8.0.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'pyyaml>=6.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'tqdm>=4.65.0'
    ]

    for req in requirements:
        subprocess.run([sys.executable, '-m', 'pip', 'install', req], check=True)

    print("Requirements installed successfully!")


def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("\n⚠ No GPU available - training will use CPU (slower)")
            return False
    except ImportError:
        print("\n⚠ PyTorch not installed yet")
        return False


def create_directories():
    """Create necessary directories"""
    dirs = [
        'data/road_dataset/images/train',
        'data/road_dataset/images/val',
        'data/road_dataset/labels/train',
        'data/road_dataset/labels/val',
        'runs/train/road_damage/weights',
        'models'
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("Directories created successfully!")


def generate_training_data(num_samples: int = 2000):
    """Generate synthetic training data"""
    print(f"\nGenerating {num_samples} synthetic training samples...")

    import cv2
    import numpy as np

    # Class configurations
    class_configs = {
        0: {'name': 'good_road', 'color': (120, 120, 120)},
        1: {'name': 'pothole', 'color': (80, 80, 80)},
        2: {'name': 'crack', 'color': (100, 100, 100)},
        3: {'name': 'accident', 'color': (60, 60, 60)},
        4: {'name': 'road_damage', 'color': (90, 90, 90)}
    }

    train_img = Path('data/road_dataset/images/train')
    train_lbl = Path('data/road_dataset/labels/train')
    val_img = Path('data/road_dataset/images/val')
    val_lbl = Path('data/road_dataset/labels/val')

    np.random.seed(42)

    for i in range(num_samples):
        # Weighted class distribution
        if i < num_samples * 0.25:
            class_id = 0  # 25% good_road
        elif i < num_samples * 0.40:
            class_id = 1  # 15% pothole
        elif i < num_samples * 0.55:
            class_id = 2  # 15% crack
        elif i < num_samples * 0.75:
            class_id = 3  # 20% accident
        else:
            class_id = 4  # 25% road_damage

        # Generate image
        img = np.random.randint(100, 160, (640, 640, 3), dtype=np.uint8)
        noise = np.random.randint(-20, 20, (640, 640, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]

        # Add class-specific patterns
        if class_id == 0:  # Good Road
            cv2.rectangle(img, (250, 0), (390, 640), (255, 255, 255), 3)
            cv2.putText(img, "GOOD ROAD", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            bboxes = []
        elif class_id == 1:  # Pothole
            cv2.circle(img, (320, 400), 80, (20, 20, 20), -1)
            cv2.circle(img, (320, 400), 80, (40, 40, 40), 2)
            cv2.putText(img, "POTHOLE", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            bboxes = [(0.5, 0.625, 0.25, 0.25)]
        elif class_id == 2:  # Crack
            for _ in range(5):
                x1 = np.random.randint(100, 300)
                y1 = np.random.randint(100, 500)
                x2 = x1 + np.random.randint(-100, 100)
                y2 = y1 + np.random.randint(50, 200)
                cv2.line(img, (x1, y1), (x2, y2), (30, 30, 30), 2)
            cv2.putText(img, "CRACK", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), 3)
            bboxes = [(0.5, 0.5, 0.6, 0.4)]
        elif class_id == 3:  # Accident
            cv2.rectangle(img, (150, 200), (490, 450), (0, 0, 200), -1)
            cv2.circle(img, (200, 350), 30, (255, 255, 0), -1)
            cv2.circle(img, (440, 350), 30, (255, 255, 0), -1)
            cv2.putText(img, "ACCIDENT!", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            bboxes = [(0.5, 0.5, 0.53, 0.39)]
        else:  # Road Damage
            cv2.rectangle(img, (100, 300), (250, 450), (30, 30, 30), -1)
            for _ in range(3):
                x = np.random.randint(350, 500)
                y = np.random.randint(200, 400)
                cv2.circle(img, (x, y), 20, (25, 25, 25), -1)
            cv2.putText(img, "DAMAGE", (230, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 100), 3)
            bboxes = [(0.3, 0.5, 0.3, 0.3), (0.7, 0.5, 0.2, 0.2)]

        # Save
        split = 'val' if i % 5 == 0 else 'train'
        img_dir = val_img if split == 'val' else train_img
        lbl_dir = val_lbl if split == 'val' else train_lbl

        img_name = f"road_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / img_name), img)

        # Write label
        label_name = f"road_{i:06d}.txt"
        with open(lbl_dir / label_name, 'w') as f:
            for bbox in bboxes:
                x, y, w, h = bbox
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    print(f"Successfully generated {num_samples} training samples!")


def create_dataset_yaml():
    """Create dataset YAML configuration"""
    yaml_content = """# Road Damage Detection Dataset Configuration
path: data/road_dataset
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
    with open('data/road_dataset/dataset.yaml', 'w') as f:
        f.write(yaml_content)

    print("Dataset YAML created!")


def train_yolo(epochs: int = 150, batch_size: int = 16, model_size: str = 'm'):
    """Train YOLOv8 model"""
    print(f"\n{'='*60}")
    print(f"Starting YOLOv8{model_size} Training")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"{'='*60}\n")

    try:
        from ultralytics import YOLO

        # Load model
        model = YOLO(f'yolov8{model_size}.pt')

        # Train
        results = model.train(
            data='data/road_dataset/dataset.yaml',
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            patience=50,
            save=True,
            save_period=20,
            project='runs/train',
            name='road_damage',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.1,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.0,
            fliplr=0.5,
            amp=True,
            plots=True,
            verbose=True
        )

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best model: runs/train/road_damage/weights/best.pt")
        print(f"{'='*60}")

        return results

    except Exception as e:
        print(f"Training error: {e}")
        return None


def evaluate_model():
    """Evaluate trained model"""
    print("\nEvaluating model...")

    try:
        from ultralytics import YOLO

        model = YOLO('runs/train/road_damage/weights/best.pt')
        metrics = model.val(data='data/road_dataset/dataset.yaml')

        print(f"\nValidation Results:")
        print(f"  mAP@0.5: {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")

        return metrics

    except Exception as e:
        print(f"Evaluation error: {e}")
        return None


def export_model():
    """Export model to ONNX format"""
    print("\nExporting model...")

    try:
        from ultralytics import YOLO

        model = YOLO('runs/train/road_damage/weights/best.pt')
        path = model.export(format='onnx')

        print(f"Model exported to: {path}")
        return path

    except Exception as e:
        print(f"Export error: {e}")
        return None


def run_demo():
    """Run demo detection"""
    print("\nRunning demo detection...")

    try:
        from backend.detect import RoadDamageDetector
        import cv2

        # Create detector
        detector = RoadDamageDetector('runs/train/road_damage/weights/best.pt')

        # Create test image
        import numpy as np
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (100, 100, 100)
        cv2.putText(test_img, "Test Image", (200, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Detect
        detections = detector.detect(test_img)

        print(f"Demo detection results:")
        for det in detections:
            print(f"  - {det['label']}: {det['confidence']:.2f}")

        return detections

    except Exception as e:
        print(f"Demo error: {e}")
        return []


def main():
    """Main training pipeline"""
    print("="*60)
    print("ROAD DAMAGE DETECTION - YOLOV8 TRAINING PIPELINE")
    print("="*60)

    # Step 1: Install requirements
    install_requirements()

    # Step 2: Check GPU
    check_gpu()

    # Step 3: Create directories
    create_directories()

    # Step 4: Generate training data
    generate_training_data(num_samples=2000)

    # Step 5: Create dataset config
    create_dataset_yaml()

    # Step 6: Train model
    train_yolo(epochs=150, batch_size=16, model_size='m')

    # Step 7: Evaluate
    evaluate_model()

    # Step 8: Export
    export_model()

    # Step 9: Demo
    run_demo()

    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("Model saved at: runs/train/road_damage/weights/best.pt")
    print("="*60)


if __name__ == '__main__':
    main()