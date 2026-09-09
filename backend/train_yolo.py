"""
YOLOv8 Road Damage Detection Training Pipeline
High-accuracy model training for road condition classification
"""
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Ultralytics YOLO
try:
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.callbacks import DefaultCallbacks
except ImportError:
    print("Installing ultralytics...")
    os.system('pip install ultralytics')
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.callbacks import DefaultCallbacks


class RoadDamageTrainer:
    """High-accuracy YOLOv8 trainer for road damage detection"""

    # Model configurations for different accuracy levels
    MODEL_CONFIGS = {
        'nano': {'model': 'yolov8n.pt', 'img_size': 640, 'epochs': 100},
        'small': {'model': 'yolov8s.pt', 'img_size': 640, 'epochs': 150},
        'medium': {'model': 'yolov8m.pt', 'img_size': 640, 'epochs': 200},
        'large': {'model': 'yolov8l.pt', 'img_size': 640, 'epochs': 250},
        'xlarge': {'model': 'yolov8x.pt', 'img_size': 640, 'epochs': 300}
    }

    def __init__(
        self,
        model_size: str = 'medium',
        data_config: str = 'data/road_dataset.yaml',
        output_dir: str = 'runs/train'
    ):
        self.model_size = model_size
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.model = None
        self.config = self.MODEL_CONFIGS.get(model_size, self.MODEL_CONFIGS['medium'])

    def create_dataset_yaml(self, dataset_path: str) -> str:
        """Create YAML configuration for dataset"""
        yaml_content = f"""# Road Damage Detection Dataset Configuration
path: {dataset_path}
train: images/train
val: images/val

# Classes - 5 detection classes
names:
  0: good_road        # Normal road surface
  1: pothole         # Potholes and cavities
  2: crack          # Surface cracks
  3: accident       # Road accidents
  4: road_damage    # General road damage

# Number of classes
nc: 5
"""
        yaml_path = Path(dataset_path).parent / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"Created dataset config: {yaml_path}")
        return str(yaml_path)

    def create_synthetic_dataset(
        self,
        num_samples: int = 1000,
        output_dir: str = 'data/road_dataset'
    ) -> str:
        """Create synthetic dataset for initial training"""
        import cv2

        output_path = Path(output_dir)
        train_img = output_path / 'images' / 'train'
        train_lbl = output_path / 'labels' / 'train'
        val_img = output_path / 'images' / 'val'
        val_lbl = output_path / 'labels' / 'val'

        for d in [train_img, train_lbl, val_img, val_lbl]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"Creating {num_samples} synthetic training samples...")

        # Class configurations
        class_configs = {
            0: {'name': 'good_road', 'color': (120, 120, 120), 'pattern': 'flat'},
            1: {'name': 'pothole', 'color': (80, 80, 80), 'pattern': 'circle'},
            2: {'name': 'crack', 'color': (100, 100, 100), 'pattern': 'line'},
            3: {'name': 'accident', 'color': (60, 60, 60), 'pattern': 'complex'},
            4: {'name': 'road_damage', 'color': (90, 90, 90), 'pattern': 'mixed'}
        }

        np.random.seed(42)
        samples_created = 0

        for i in range(num_samples):
            # Determine class (weighted - more good roads)
            if i < num_samples * 0.3:
                class_id = 0  # good_road
            elif i < num_samples * 0.5:
                class_id = 1  # pothole
            elif i < num_samples * 0.65:
                class_id = 2  # crack
            elif i < num_samples * 0.8:
                class_id = 3  # accident
            else:
                class_id = 4  # road_damage

            cfg = class_configs[class_id]
            img = self._generate_synthetic_image(class_id, cfg)

            # Split: 80% train, 20% val
            if i % 5 == 0:
                img_dir, lbl_dir = val_img, val_lbl
            else:
                img_dir, lbl_dir = train_img, train_lbl

            # Save image
            img_name = f"road_{i:06d}.jpg"
            cv2.imwrite(str(img_dir / img_name), img)

            # Save label
            label_name = f"road_{i:06d}.txt"
            self._write_label(train_lbl if i % 5 != 0 else val_lbl, label_name, class_id, cfg)

            samples_created += 1

            if (i + 1) % 200 == 0:
                print(f"  Created {i + 1}/{num_samples} samples...")

        print(f"Created {samples_created} synthetic samples")
        return str(output_path)

    def _generate_synthetic_image(self, class_id: int, cfg: Dict) -> np.ndarray:
        """Generate synthetic road image"""
        import cv2

        img = np.random.randint(100, 160, (640, 640, 3), dtype=np.uint8)

        # Add road texture
        noise = np.random.randint(-20, 20, (640, 640, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]

        if class_id == 0:  # good_road
            # Clean road surface with lane markings
            cv2.rectangle(img, (250, 0), (390, 640), (255, 255, 255), 3)
            cv2.putText(img, "GOOD ROAD", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        elif class_id == 1:  # pothole
            # Add pothole (dark circle)
            cx, cy = 320, 400
            cv2.circle(img, (cx, cy), 80, (20, 20, 20), -1)
            cv2.circle(img, (cx, cy), 80, (40, 40, 40), 2)
            cv2.putText(img, "POTHOLE", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        elif class_id == 2:  # crack
            # Add crack lines
            for _ in range(5):
                x1 = np.random.randint(100, 300)
                y1 = np.random.randint(100, 500)
                x2 = x1 + np.random.randint(-100, 100)
                y2 = y1 + np.random.randint(50, 200)
                cv2.line(img, (x1, y1), (x2, y2), (30, 30, 30), 2)
            cv2.putText(img, "CRACK", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), 3)

        elif class_id == 3:  # accident
            # Add accident indicators
            cv2.rectangle(img, (150, 200), (490, 450), (0, 0, 200), -1)
            cv2.circle(img, (200, 350), 30, (255, 255, 0), -1)
            cv2.circle(img, (440, 350), 30, (255, 255, 0), -1)
            cv2.putText(img, "ACCIDENT!", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        else:  # road_damage
            # Mixed damage
            cv2.rectangle(img, (100, 300), (250, 450), (30, 30, 30), -1)
            for _ in range(3):
                x = np.random.randint(350, 500)
                y = np.random.randint(200, 400)
                cv2.circle(img, (x, y), 20, (25, 25, 25), -1)
            cv2.putText(img, "DAMAGE", (230, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 100), 3)

        return img

    def _write_label(self, label_dir: Path, label_name: str, class_id: int, cfg: Dict):
        """Write YOLO format label"""
        # Normalized coordinates for 640x640 image
        bboxes = {
            0: [],  # good_road - no objects
            1: [(0.5, 0.625, 0.25, 0.25)],  # pothole
            2: [(0.5, 0.5, 0.6, 0.4)],  # crack
            3: [(0.5, 0.5, 0.53, 0.39)],  # accident
            4: [(0.3, 0.5, 0.3, 0.3), (0.7, 0.5, 0.2, 0.2)]  # road_damage
        }

        with open(label_dir / label_name, 'w') as f:
            for bbox in bboxes[class_id]:
                x, y, w, h = bbox
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: int = 16,
        patience: int = 50,
        save_period: int = 10,
        pretrained: bool = True,
        augment: bool = True,
        verbose: bool = True
    ) -> str:
        """Train the YOLOv8 model with high accuracy settings"""

        epochs = epochs or self.config['epochs']
        model_name = self.config['model']

        print(f"\n{'='*60}")
        print(f"Training YOLOv8-{self.model_size.upper()} for Road Damage Detection")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {self.config['img_size']}")
        print(f"{'='*60}\n")

        # Initialize model
        self.model = YOLO(model_name)

        # Training parameters optimized for high accuracy
        results = self.model.train(
            data=self.data_config,
            epochs=epochs,
            imgsz=self.config['img_size'],
            batch=batch_size,
            patience=patience,
            save=save_period > 0,
            save_period=save_period,
            project=str(self.output_dir),
            name='road_damage',
            exist_ok=True,
            pretrained=pretrained,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
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
            perspective=0.0001,
            flipud=0.0,
            fliplr=0.5,
            amp=True,
            plots=True,
            verbose=verbose
        )

        # Get best model path
        best_model = self.output_dir / 'road_damage' / 'weights' / 'best.pt'
        if not best_model.exists():
            best_model = self.output_dir / 'road_damage' / 'weights' / 'last.pt'

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best model saved at: {best_model}")
        print(f"{'='*60}")

        return str(best_model)

    def validate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate the trained model"""
        if model_path:
            self.model = YOLO(model_path)

        if not self.model:
            print("No model loaded. Please train first or provide model path.")
            return {}

        print("\nValidating model...")
        metrics = self.model.val(
            data=self.data_config,
            split='val',
            imgsz=self.config['img_size'],
            batch=batch_size
        )

        return {
            'map50': metrics.box.map50,
            'map50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr
        }

    def export_model(self, model_path: str, format: str = 'onnx') -> str:
        """Export model to different formats"""
        model = YOLO(model_path)
        export_path = model.export(format=format)
        return export_path


def train_accident_classifier(
    data_path: str = 'data/road_dataset',
    model_size: str = 'medium',
    high_accuracy: bool = True
) -> str:
    """Train high-accuracy accident classification model"""

    trainer = RoadDamageTrainer(
        model_size=model_size,
        data_config=f'{data_path}/dataset.yaml',
        output_dir='runs/train'
    )

    # Create synthetic dataset if needed
    if not Path(data_path).exists():
        print("Creating synthetic dataset...")
        data_path = trainer.create_synthetic_dataset(num_samples=2000)

    # Create YAML config
    yaml_config = trainer.create_dataset_yaml(data_path)

    # Train with optimized parameters
    epochs = 200 if high_accuracy else 100
    batch_size = 16 if high_accuracy else 8

    best_model = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        patience=50 if high_accuracy else 30,
        save_period=20,
        pretrained=True,
        augment=True
    )

    return best_model


# Training utilities
def load_training_data(data_dir: str):
    """Load training data statistics"""
    from pathlib import Path

    data_path = Path(data_dir)
    if not data_path.exists():
        return None

    train_images = len(list((data_path / 'images' / 'train').glob('*.jpg')))
    val_images = len(list((data_path / 'images' / 'val').glob('*.jpg')))

    return {
        'train_samples': train_images,
        'val_samples': val_images,
        'total_samples': train_images + val_images
    }


def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("No GPU available - training will use CPU")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLOv8 Road Damage Detector')
    parser.add_argument('--model', default='medium', choices=['nano', 'small', 'medium', 'large', 'xlarge'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--data', default='data/road_dataset')
    parser.add_argument('--synthetic', action='store_true', help='Create synthetic dataset')

    args = parser.parse_args()

    # Check GPU
    check_gpu()

    # Initialize trainer
    trainer = RoadDamageTrainer(
        model_size=args.model,
        data_config=f'{args.data}/dataset.yaml'
    )

    # Create synthetic data if requested
    if args.synthetic:
        args.data = trainer.create_synthetic_dataset(num_samples=2000)
        trainer.data_config = trainer.create_dataset_yaml(args.data)

    # Train
    best_model = trainer.train(epochs=args.epochs, batch_size=args.batch)

    print(f"\nTraining complete!")
    print(f"Model saved at: {best_model}")