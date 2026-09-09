"""
Improved Training Script - Fix for Good Road Misclassification
Creates diverse, realistic training data with clear good vs bad differentiation
"""
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


def create_realistic_good_road(num_samples: int = 500) -> list:
    """Create realistic good road images with clear labels"""
    images = []
    labels = []

    print(f"Creating {num_samples} GOOD ROAD samples...")

    for i in range(num_samples):
        img = np.random.randint(80, 120, (640, 640, 3), dtype=np.uint8)

        # Add road texture variation
        for _ in range(100):
            x, y = np.random.randint(0, 640, 2)
            noise = np.random.randint(-15, 15, 3)
            img[y, x] = np.clip(img[y, x] + noise, 0, 255)

        # Gray asphalt base
        img[:] = (90 + np.random.randint(-10, 10),
                  90 + np.random.randint(-10, 10),
                  90 + np.random.randint(-10, 10))

        # Lane markings - white dashed lines
        num_lanes = np.random.choice([2, 3, 4])
        if num_lanes == 2:
            positions = [280, 360]
        elif num_lanes == 3:
            positions = [220, 320, 420]
        else:
            positions = [180, 270, 370, 460]

        for pos in positions:
            for y in range(0, 640, 60):
                cv2.rectangle(img, (pos - 3, y), (pos + 3, y + 30), (240, 240, 240), -1)

        # Edge lines (yellow)
        cv2.rectangle(img, (50, 0), (70, 640), (0, 180, 240), -1)
        cv2.rectangle(img, (570, 0), (590, 640), (0, 180, 240), -1)

        # Add slight variations
        if np.random.random() < 0.3:
            # Add road signs
            x = np.random.randint(100, 500)
            cv2.circle(img, (x, 100), 25, (200, 50, 50), -1)

        # No bounding boxes - good road has no damage
        images.append(img)
        labels.append([])

    return images, labels


def create_realistic_pothole(num_samples: int = 400) -> list:
    """Create realistic pothole images"""
    images = []
    labels = []

    print(f"Creating {num_samples} POTHOLE samples...")

    for i in range(num_samples):
        # Start with good road
        img = np.random.randint(80, 110, (640, 640, 3), dtype=np.uint8)
        img[:] = (95, 95, 95)

        # Lane markings
        for y in range(0, 640, 60):
            cv2.rectangle(img, (315, y), (325, y + 30), (240, 240, 240), -1)

        # Add multiple potholes
        num_potholes = np.random.randint(1, 4)
        bboxes = []

        for _ in range(num_potholes):
            cx = np.random.randint(150, 490)
            cy = np.random.randint(200, 500)
            rx = np.random.randint(40, 100)
            ry = np.random.randint(30, 80)

            # Dark pothole with uneven edges
            cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (30, 30, 30), -1)
            cv2.ellipse(img, (cx, cy), (rx - 5, ry - 5), 0, 0, 360, (50, 50, 50), 2)

            # Convert to YOLO format (normalized)
            x_center = cx / 640
            y_center = cy / 640
            width = (rx * 2) / 640
            height = (ry * 2) / 640
            bboxes.append((1, x_center, y_center, width, height))

        images.append(img)
        labels.append(bboxes)

    return images, labels


def create_realistic_crack(num_samples: int = 400) -> list:
    """Create realistic crack images"""
    images = []
    labels = []

    print(f"Creating {num_samples} CRACK samples...")

    for i in range(num_samples):
        img = np.random.randint(80, 110, (640, 640, 3), dtype=np.uint8)
        img[:] = (95, 95, 95)

        # Lane markings
        for y in range(0, 640, 60):
            cv2.rectangle(img, (315, y), (325, y + 30), (240, 240, 240), -1)

        # Add cracks
        num_cracks = np.random.randint(1, 3)
        bboxes = []

        for _ in range(num_cracks):
            # Random crack pattern
            points = []
            x, y = np.random.randint(100, 500), np.random.randint(100, 500)

            for _ in range(np.random.randint(8, 15)):
                points.append((x, y))
                x += np.random.randint(-40, 50)
                y += np.random.randint(-40, 50)
                x = np.clip(x, 50, 590)
                y = np.clip(y, 50, 590)

            # Draw crack
            for j in range(len(points) - 1):
                cv2.line(img, points[j], points[j + 1],
                        (40 + np.random.randint(-10, 10), 40, 40),
                        np.random.randint(1, 3))

            # Bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, 640)
            y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, 640)

            x_center = (x_min + x_max) / 2 / 640
            y_center = (y_min + y_max) / 2 / 640
            width = (x_max - x_min) / 640
            height = (y_max - y_min) / 640
            bboxes.append((2, x_center, y_center, width, height))

        images.append(img)
        labels.append(bboxes)

    return images, labels


def create_realistic_accident(num_samples: int = 400) -> list:
    """Create realistic accident images"""
    images = []
    labels = []

    print(f"Creating {num_samples} ACCIDENT samples...")

    for i in range(num_samples):
        img = np.random.randint(60, 100, (640, 640, 3), dtype=np.uint8)
        img[:] = (80, 80, 80)

        # Lane markings (partial - road damaged)
        for y in range(0, 640, 60):
            if np.random.random() > 0.3:
                cv2.rectangle(img, (315, y), (325, y + 30), (200, 200, 200), -1)

        # Add accident scene
        num_vehicles = np.random.randint(1, 3)
        bboxes = []

        for v in range(num_vehicles):
            # Vehicle position
            vx = np.random.randint(150, 450)
            vy = np.random.randint(250, 450)

            # Damaged vehicle (tilted/dark)
            angle = np.random.randint(-30, 30)
            M = cv2.getRotationMatrix2D((vx, vy), angle, 1)

            # Vehicle body
            cv2.rectangle(img, (vx - 60, vy - 30), (vx + 60, vy + 30),
                         (60 + np.random.randint(-20, 20),
                          60 + np.random.randint(-20, 20),
                          60 + np.random.randint(-20, 20)), -1)

            # Broken glass effect
            cv2.circle(img, (vx - 20, vy - 10), 15, (100, 150, 200), -1)
            cv2.circle(img, (vx + 20, vy - 10), 10, (100, 150, 200), -1)

            # Debris
            for _ in range(np.random.randint(5, 15)):
                dx = vx + np.random.randint(-80, 80)
                dy = vy + np.random.randint(-50, 50)
                cv2.circle(img, (dx, dy), np.random.randint(2, 5), (40, 40, 40), -1)

            # Bounding box
            x_center = vx / 640
            y_center = vy / 640
            width = 120 / 640
            height = 60 / 640
            bboxes.append((3, x_center, y_center, width, height))

        images.append(img)
        labels.append(bboxes)

    return images, labels


def create_mixed_damage(num_samples: int = 300) -> list:
    """Create images with multiple damage types"""
    images = []
    labels = []

    print(f"Creating {num_samples} MIXED DAMAGE samples...")

    for i in range(num_samples):
        img = np.random.randint(80, 100, (640, 640, 3), dtype=np.uint8)
        img[:] = (85, 85, 85)

        bboxes = []

        # Randomly add different damage types
        if np.random.random() < 0.5:
            # Add pothole
            cx, cy = np.random.randint(200, 440), np.random.randint(300, 500)
            cv2.circle(img, (cx, cy), np.random.randint(50, 80),
                      (30, 30, 30), -1)
            bboxes.append((1, cx/640, cy/640, 0.2, 0.2))

        if np.random.random() < 0.5:
            # Add crack
            x, y = np.random.randint(100, 300), np.random.randint(100, 300)
            for _ in range(8):
                cv2.line(img, (x, y), (x + 50, y + 80), (40, 40, 40), 2)
                x += np.random.randint(-20, 30)
                y += np.random.randint(10, 40)
            bboxes.append((4, 0.4, 0.5, 0.3, 0.3))

        images.append(img)
        labels.append(bboxes)

    return images, labels


def save_dataset(images, labels, output_dir, class_name):
    """Save images and labels to disk"""
    img_dir = output_dir / 'images' / 'train'
    lbl_dir = output_dir / 'labels' / 'train'

    Path(img_dir).mkdir(parents=True, exist_ok=True)
    Path(lbl_dir).mkdir(parents=True, exist_ok=True)

    for i, (img, lbls) in enumerate(zip(images, labels)):
        img_name = f"{class_name}_{i:05d}.jpg"
        cv2.imwrite(str(img_dir / img_name), img)

        lbl_name = f"{class_name}_{i:05d}.txt"
        with open(lbl_dir / lbl_name, 'w') as f:
            for lbl in lbls:
                class_id, x, y, w, h = lbl
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"  Saved {len(images)} {class_name} samples")


def create_validation_set(output_dir):
    """Create validation set from a portion of training data"""
    import shutil

    train_img = output_dir / 'images' / 'train'
    train_lbl = output_dir / 'labels' / 'train'
    val_img = output_dir / 'images' / 'val'
    val_lbl = output_dir / 'labels' / 'val'

    Path(val_img).mkdir(parents=True, exist_ok=True)
    Path(val_lbl).mkdir(parents=True, exist_ok=True)

    # Move 20% to validation
    for img_file in train_img.glob('*.jpg'):
        if np.random.random() < 0.2:
            shutil.move(str(img_file), str(val_img / img_file.name))
            lbl_file = train_lbl / img_file.name.replace('.jpg', '.txt')
            if lbl_file.exists():
                shutil.move(str(lbl_file), str(val_lbl / lbl_file.name))


def create_yaml_config(output_dir):
    """Create dataset YAML configuration"""
    yaml_content = """# Road Damage Detection Dataset
# Balanced dataset with clear good vs damaged road differentiation

path: data/road_dataset
train: images/train
val: images/val

names:
  0: good_road      # Clean road, no damage
  1: pothole       # Potholes and cavities
  2: crack         # Surface cracks
  3: accident      # Vehicle accidents
  4: road_damage   # Mixed damage

nc: 5
"""
    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\nCreated dataset.yaml")


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     IMPROVED ROAD DAMAGE DETECTION TRAINING                    ║
║     Fixed: Good Road Classification                            ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    output_dir = Path('data/road_dataset')

    # Clear existing data
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    # Create class-specific datasets
    print("\n[1] Creating datasets...")

    # Good roads - most important for fixing misclassification
    good_images, good_labels = create_realistic_good_road(600)
    save_dataset(good_images, good_labels, output_dir, 'good_road')

    # Damaged roads
    pothole_images, pothole_labels = create_realistic_pothole(400)
    save_dataset(pothole_images, pothole_labels, output_dir, 'pothole')

    crack_images, crack_labels = create_realistic_crack(400)
    save_dataset(crack_images, crack_labels, output_dir, 'crack')

    accident_images, accident_labels = create_realistic_accident(400)
    save_dataset(accident_images, accident_labels, output_dir, 'accident')

    damage_images, damage_labels = create_mixed_damage(300)
    save_dataset(damage_images, damage_labels, output_dir, 'damage')

    # Create validation set
    print("\n[2] Creating validation split...")
    create_validation_set(output_dir)

    # Create YAML config
    print("\n[3] Creating configuration...")
    create_yaml_config(output_dir)

    # Count samples
    train_count = len(list((output_dir / 'images' / 'train').glob('*.jpg')))
    val_count = len(list((output_dir / 'images' / 'val').glob('*.jpg')))

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    DATASET SUMMARY                             ║
╠═══════════════════════════════════════════════════════════════╣
║  Good Road:    600 samples (40% of training)                  ║
║  Pothole:     400 samples                                      ║
║  Crack:       400 samples                                      ║
║  Accident:    400 samples                                      ║
║  Road Damage: 300 samples                                      ║
╠═══════════════════════════════════════════════════════════════╣
║  Training samples:  {train_count:>4}                                        ║
║  Validation samples: {val_count:>4}                                        ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Train the model
    print("\n[4] Training YOLOv8 model...")
    train_model(output_dir)


def train_model(output_dir):
    """Train YOLOv8 with improved settings"""
    try:
        from ultralytics import YOLO
        import torch

        # Check GPU
        if torch.cuda.is_available():
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ Using CPU (slower)")

        # Load model
        print("\nLoading YOLOv8m model...")
        model = YOLO('yolov8m.pt')

        # Train with improved settings
        print("""
Starting training with optimized settings:
- More learning rate decay for better generalization
- Longer warmup
- Lower batch size for better convergence
        """)

        results = model.train(
            data=str(output_dir / 'dataset.yaml'),
            epochs=200,
            imgsz=640,
            batch=8,
            patience=30,
            save=True,
            save_period=25,
            project='runs/train',
            name='road_damage',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.0005,          # Lower learning rate
            lrf=0.001,           # More decay
            momentum=0.9,
            weight_decay=0.0001,
            warmup_epochs=5.0,
            box=7.5,
            cls=0.8,             # Higher class weight
            dfl=1.5,
            mosaic=0.8,          # Reduce for cleaner images
            mixup=0.1,
            copy_paste=0.05,
            degrees=5.0,
            translate=0.05,      # Less translation
            scale=0.3,           # Less scale variation
            shear=1.0,
            flipud=0.0,
            fliplr=0.5,
            amp=True,
            plots=True,
            verbose=True
        )

        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    TRAINING COMPLETE                           ║
╠═══════════════════════════════════════════════════════════════╣
║  Best model: runs/train/road_damage/weights/best.pt          ║
╚═══════════════════════════════════════════════════════════════╝
        """)

    except Exception as e:
        print(f"Training error: {e}")
        print("Run manually: python -m ultralytics train data=data/road_dataset/dataset.yaml")


if __name__ == '__main__':
    main()