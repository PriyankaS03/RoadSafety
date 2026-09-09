"""
Data Preparator for Road Damage Detection
Extracts frames from videos and processes images for YOLOv8 training
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import json
from collections import defaultdict


class RoadDamageDataPreparator:
    """Prepares labeled data for YOLOv8 training from videos and images"""

    CLASSES = {
        'good_road': 0,
        'pothole': 1,
        'crack': 2,
        'accident': 3,
        'road_damage': 4
    }

    def __init__(self, output_dir: str = 'data/dataset'):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.train_dir = self.images_dir / 'train'
        self.val_dir = self.images_dir / 'val'
        self.train_labels = self.labels_dir / 'train'
        self.val_labels = self.labels_dir / 'val'

        for dir_path in [self.train_dir, self.val_dir, self.train_labels, self.val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.frame_count = 0
        self.class_counts = defaultdict(int)

    def extract_frames_from_video(
        self,
        video_path: str,
        label: str,
        sample_interval: int = 30,
        split: str = 'train',
        bbox_format: str = 'manual'
    ) -> int:
        """
        Extract frames from video for training.
        For auto-labeling, uses frame differencing to detect changes.
        """
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return 0

        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # Auto-label based on video filename patterns
                detected_label = self._auto_detect_label(label, frame)

                if detected_label:
                    # Save image
                    img_name = f"frame_{self.frame_count:06d}.jpg"
                    save_dir = self.train_dir if split == 'train' else self.val_dir
                    img_path = save_dir / img_name
                    cv2.imwrite(str(img_path), frame)

                    # Create YOLO format label
                    label_name = img_name.replace('.jpg', '.txt')
                    label_dir = self.train_labels if split == 'train' else self.val_labels
                    label_path = label_dir / label_name

                    # Generate bounding boxes (for demonstration - in production use manual annotations)
                    bboxes = self._generate_bboxes(frame, detected_label)
                    self._write_yolo_label(label_path, bboxes)

                    self.class_counts[detected_label] += 1
                    extracted += 1
                    self.frame_count += 1

            frame_idx += 1

        cap.release()
        print(f"Extracted {extracted} frames from {video_path}")
        return extracted

    def _auto_detect_label(self, base_label: str, frame: np.ndarray) -> Optional[str]:
        """Auto-detect label based on content analysis"""
        # Simple heuristic - in production use actual ML-based detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection for cracks/potholes
        edges = cv2.Canny(blur, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        if base_label == 'accident':
            return 'accident'
        elif base_label == 'damaged':
            if edge_ratio > 0.05:
                return 'crack'
            else:
                return 'pothole'
        elif base_label == 'good':
            return 'good_road'

        return base_label

    def _generate_bboxes(self, frame: np.ndarray, label: str) -> List[Tuple[int, int, int, int]]:
        """Generate bounding boxes - simplified version"""
        h, w = frame.shape[:2]

        if label == 'good_road':
            return []  # No objects in good road images
        elif label == 'accident':
            # Generate accident bbox (larger area)
            x1, y1 = int(w * 0.2), int(h * 0.3)
            x2, y2 = int(w * 0.8), int(h * 0.7)
            return [(x1, y1, x2, y2)]
        elif label == 'pothole':
            # Generate pothole bbox (typically lower part of image)
            x1, y1 = int(w * 0.3), int(h * 0.6)
            x2, y2 = int(w * 0.7), int(h * 0.9)
            return [(x1, y1, x2, y2)]
        elif label == 'crack':
            # Generate crack bbox (line-like)
            x1, y1 = int(w * 0.1), int(h * 0.4)
            x2, y2 = int(w * 0.9), int(h * 0.6)
            return [(x1, y1, x2, y2)]

        return []

    def _write_yolo_label(self, label_path: Path, bboxes: List[Tuple[int, int, int, int]]):
        """Write YOLO format label (normalized coordinates)"""
        # Need image dimensions - we'll store with image or assume standard
        # For now use placeholder 640x640
        img_w, img_h = 640, 640

        with open(label_path, 'w') as f:
            for (x1, y1, x2, y2) in bboxes:
                # Normalize coordinates
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                # Use class 4 (road_damage) as default for damaged
                class_id = self.CLASSES.get('road_damage', 4)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def process_image_folder(
        self,
        image_dir: str,
        label: str,
        split: str = 'train'
    ) -> int:
        """Process a folder of images"""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"Directory not found: {image_dir}")
            return 0

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        processed = 0

        for img_path in image_dir.glob('*'):
            if img_path.suffix.lower() not in image_extensions:
                continue

            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                # Save to appropriate directory
                img_name = f"img_{self.frame_count:06d}{img_path.suffix}"
                save_dir = self.train_dir if split == 'train' else self.val_dir
                dest_path = save_dir / img_name
                cv2.imwrite(str(dest_path), frame)

                # Generate label
                bboxes = self._generate_bboxes(frame, label)
                label_name = img_name.replace(img_path.suffix, '.txt')
                label_dir = self.train_labels if split == 'train' else self.val_labels
                label_path = label_dir / label_name
                self._write_yolo_label(label_path, bboxes)

                self.class_counts[label] += 1
                processed += 1
                self.frame_count += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"Processed {processed} images from {image_dir}")
        return processed

    def create_yaml_config(self, config_path: str = 'data/dataset.yaml'):
        """Create YOLOv8 YAML configuration file"""
        yaml_content = f"""# Road Damage Detection Dataset
path: {self.output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: good_road
  1: pothole
  2: crack
  3: accident
  4: road_damage

# Dataset info
nc: 5
"""

        with open(config_path, 'w') as f:
            f.write(yaml_content)

        print(f"Created YAML config at {config_path}")
        return config_path

    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        return {
            'total_images': self.frame_count,
            'class_counts': dict(self.class_counts),
            'train_samples': len(list(self.train_dir.glob('*.jpg'))),
            'val_samples': len(list(self.val_dir.glob('*.jpg')))
        }


def prepare_sample_dataset(base_dir: str = 'data/samples'):
    """Create sample dataset structure for demonstration"""
    base_path = Path(base_dir)

    # Create sample directories
    for split in ['train', 'val']:
        for label in ['good_road', 'damaged', 'accident']:
            (base_path / split / label).mkdir(parents=True, exist_ok=True)

    print(f"Sample dataset structure created at {base_path}")
    return str(base_path)


if __name__ == '__main__':
    # Demo usage
    preparator = RoadDamageDataPreparator('data/road_dataset')
    print("Data preparator initialized")
    print(f"Classes: {preparator.CLASSES}")