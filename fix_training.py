"""
Quick Fix: Regenerate Dataset with Proper Good vs Bad Road Differentiation
Run this to fix the misclassification issue
"""
import cv2
import numpy as np
from pathlib import Path
import shutil


def clean_and_create_dataset():
    """Create properly balanced dataset"""

    # Clean old data
    data_dir = Path('data/road_dataset')
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # Create directories
    train_img = data_dir / 'images' / 'train'
    val_img = data_dir / 'images' / 'val'
    train_lbl = data_dir / 'labels' / 'train'
    val_lbl = data_dir / 'labels' / 'val'

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    print("Creating dataset with clear GOOD vs BAD road differentiation...\n")

    sample_count = 0
    np.random.seed(42)

    # ============================================
    # GOOD ROAD - Clean, normal road surfaces
    # ============================================
    print("Creating GOOD ROAD samples (no damage)...")

    for i in range(800):  # More good road samples
        img = np.zeros((640, 640, 3), dtype=np.uint8)

        # Asphalt base - vary the gray
        gray_val = np.random.randint(85, 105)
        img[:] = (gray_val, gray_val, gray_val)

        # Add subtle texture
        noise = np.random.randint(-5, 5, (640, 640, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # White lane markings (dashed)
        for y in range(0, 640, 50):
            cv2.rectangle(img, (310, y), (330, y + 25), (230, 230, 230), -1)

        # Yellow edge lines
        cv2.rectangle(img, (40, 0), (55, 640), (0, 170, 230), -1)
        cv2.rectangle(img, (585, 0), (600, 640), (0, 170, 230), -1)

        # Save
        if i % 5 == 0:
            cv2.imwrite(str(val_img / f"good_{i:05d}.jpg"), img)
            with open(val_lbl / f"good_{i:05d}.txt", 'w') as f:
                pass  # No objects = empty label
        else:
            cv2.imwrite(str(train_img / f"good_{i:05d}.jpg"), img)
            with open(train_lbl / f"good_{i:05d}.txt", 'w') as f:
                pass  # No objects = empty label

        sample_count += 1

    print(f"  Created {sample_count} good road samples")

    # ============================================
    # POTHOLE - Clear dark circles/irregular shapes
    # ============================================
    print("Creating POTHOLE samples...")

    for i in range(400):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        gray_val = np.random.randint(85, 100)
        img[:] = (gray_val, gray_val, gray_val)

        # Lane markings
        for y in range(0, 640, 50):
            cv2.rectangle(img, (310, y), (330, y + 25), (230, 230, 230), -1)

        # Add pothole(s)
        num_potholes = np.random.randint(1, 3)
        bboxes = []

        for _ in range(num_potholes):
            cx = np.random.randint(150, 490)
            cy = np.random.randint(250, 500)
            rx = np.random.randint(40, 90)

            # Dark irregular shape
            cv2.circle(img, (cx, cy), rx, (20, 20, 20), -1)
            cv2.circle(img, (cx, cy), rx + 5, (40, 40, 40), 2)

            # Bbox
            x, y, w, h = (cx - rx) / 640, (cy - rx) / 640, (rx * 2) / 640, (rx * 2) / 640
            bboxes.append(f"1 {x + w/2:.4f} {y + h/2:.4f} {w:.4f} {h:.4f}")

        # Save
        if i % 5 == 0:
            cv2.imwrite(str(val_img / f"pothole_{i:05d}.jpg"), img)
            with open(val_lbl / f"pothole_{i:05d}.txt", 'w') as f:
                f.write('\n'.join(bboxes))
        else:
            cv2.imwrite(str(train_img / f"pothole_{i:05d}.jpg"), img)
            with open(train_lbl / f"pothole_{i:05d}.txt", 'w') as f:
                f.write('\n'.join(bboxes))

    print(f"  Created 400 pothole samples")

    # ============================================
    # CRACK - Lines and cracks
    # ============================================
    print("Creating CRACK samples...")

    for i in range(400):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        gray_val = np.random.randint(85, 100)
        img[:] = (gray_val, gray_val, gray_val)

        # Lane markings
        for y in range(0, 640, 50):
            cv2.rectangle(img, (310, y), (330, y + 25), (230, 230, 230), -1)

        # Add cracks
        bboxes = []
        start_x = np.random.randint(100, 400)
        start_y = np.random.randint(100, 400)

        for j in range(6):
            x = start_x + j * np.random.randint(20, 60)
            y = start_y + j * np.random.randint(-30, 30)
            x = np.clip(x, 100, 540)
            y = np.clip(y, 100, 540)

            cv2.line(img, (x, y), (x + 40, y + np.random.randint(-20, 20)),
                    (35, 35, 35), np.random.randint(2, 4))

        # Box around crack area
        bboxes.append(f"2 0.5 0.5 0.6 0.5")

        # Save
        if i % 5 == 0:
            cv2.imwrite(str(val_img / f"crack_{i:05d}.jpg"), img)
            with open(val_lbl / f"crack_{i:05d}.txt", 'w') as f:
                f.write('\n'.join(bboxes))
        else:
            cv2.imwrite(str(train_img / f"crack_{i:05d}.jpg"), img)
            with open(train_lbl / f"crack_{i:05d}.txt", 'w') as f:
                f.write('\n'.join(bboxes))

    print(f"  Created 400 crack samples")

    # ============================================
    # ACCIDENT - Damaged vehicles
    # ============================================
    print("Creating ACCIDENT samples...")

    for i in range(400):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        gray_val = np.random.randint(75, 90)
        img[:] = (gray_val, gray_val, gray_val)

        # Faded lane markings
        for y in range(0, 640, 50):
            if np.random.random() > 0.3:
                cv2.rectangle(img, (310, y), (330, y + 25), (180, 180, 180), -1)

        # Damaged vehicle
        vx = np.random.randint(200, 440)
        vy = np.random.randint(250, 400)

        cv2.rectangle(img, (vx - 70, vy - 35), (vx + 70, vy + 35),
                     (50, 50, 50), -1)
        cv2.circle(img, (vx - 25, vy), 15, (80, 100, 120), -1)
        cv2.circle(img, (vx + 25, vy), 15, (80, 100, 120), -1)

        # Debris
        for _ in range(8):
            dx = vx + np.random.randint(-90, 90)
            dy = vy + np.random.randint(-50, 50)
            cv2.circle(img, (dx, dy), np.random.randint(2, 6), (30, 30, 30), -1)

        bbox = f"3 {vx/640:.4f} {vy/640:.4f} 0.22 0.22"

        # Save
        if i % 5 == 0:
            cv2.imwrite(str(val_img / f"accident_{i:05d}.jpg"), img)
            with open(val_lbl / f"accident_{i:05d}.txt", 'w') as f:
                f.write(bbox)
        else:
            cv2.imwrite(str(train_img / f"accident_{i:05d}.jpg"), img)
            with open(train_lbl / f"accident_{i:05d}.txt", 'w') as f:
                f.write(bbox)

    print(f"  Created 400 accident samples")

    # ============================================
    # Create YAML config
    # ============================================
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
    with open(data_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)

    # Summary
    train_count = len(list(train_img.glob('*.jpg')))
    val_count = len(list(val_img.glob('*.jpg')))

    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                     DATASET CREATED                             ║
╠═══════════════════════════════════════════════════════════════════╣
║  Good Road:    800 samples (NO bounding boxes)                   ║
║  Pothole:     400 samples                                        ║
║  Crack:       400 samples                                        ║
║  Accident:    400 samples                                        ║
╠═══════════════════════════════════════════════════════════════════╣
║  Training:   {train_count:>4} samples                                       ║
║  Validation: {val_count:>4} samples                                       ║
╚═══════════════════════════════════════════════════════════════════╝

Key fix: Good road images have NO bounding boxes (empty labels)
This teaches the model that absence of boxes = good road

Now training...
    """)

    return data_dir


def train():
    """Train the model"""
    try:
        from ultralytics import YOLO
        import torch

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        print("\nTraining YOLOv8m...")
        model = YOLO('yolov8m.pt')

        results = model.train(
            data='data/road_dataset/dataset.yaml',
            epochs=150,
            imgsz=640,
            batch=16,
            patience=40,
            save=True,
            project='runs/train',
            name='road_damage',
            exist_ok=True,
            optimizer='AdamW',
            lr0=0.001,
            cls=1.0,  # Higher class weight for better classification
            amp=True,
            plots=True
        )

        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    TRAINING COMPLETE!                             ║
╠═══════════════════════════════════════════════════════════════════╣
║  Model saved: runs/train/road_damage/weights/best.pt             ║
╚═══════════════════════════════════════════════════════════════════╝
        """)

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == '__main__':
    data_dir = clean_and_create_dataset()
    train()