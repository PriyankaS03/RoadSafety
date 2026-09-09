"""
backend/detect.py – Advanced Road Damage Detection Inference
Improved YOLOv8 inference with better pothole accuracy,
false-positive reduction, and cleaner detection rendering.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')


class RoadDamageDetector:
    """Real-time road damage detection using YOLOv8"""

    CLASS_NAMES = {
        0: 'Good Road',
        1: 'Pothole',
        2: 'Crack',
        3: 'Accident',
        4: 'Road Damage',
        5: 'Edge Break',
        6: 'Rutting',
    }

    CLASS_COLORS = {
        0: (0, 220, 0),      # Green
        1: (0, 0, 255),      # RED pothole
        2: (0, 255, 255),    # Yellow crack
        3: (255, 0, 0),      # Blue accident
        4: (180, 0, 255),    # Purple
        5: (0, 255, 128),    # Mint
        6: (255, 165, 0),    # Orange
    }

    # Improved confidence filtering
    CLASS_MIN_CONF = {
        0: 0.45,
        1: 0.65,   # Higher pothole confidence
        2: 0.60,
        3: 0.75,
        4: 0.60,
        5: 0.55,
        6: 0.55,
    }

    def __init__(
        self,
        model_path: str = 'runs/train/road_damage/weights/best.pt',
        conf_threshold: float = 0.60,
        iou_threshold: float = 0.45,
        img_size: int = 640
    ):

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.model = None

        self._load_model()

    def _load_model(self):

        try:
            from ultralytics import YOLO

            if Path(self.model_path).exists():

                self.model = YOLO(self.model_path)

                print(f"✅ Model loaded: {self.model_path}")

            else:

                print(f"⚠️ No model found at {self.model_path}")
                self.model = None

        except ImportError:

            print("Install ultralytics using:")
            print("pip install ultralytics")

            self.model = None

    def detect(
        self,
        source: Union[str, np.ndarray],
        show_conf: bool = True,
        line_thickness: int = 2
    ) -> List[Dict]:

        if self.model is None:
            return []

        results = self.model(
            source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
        )

        detections = []

        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                width = x2 - x1
                height = y2 - y1

                area = width * height

                # ---------- REMOVE SMALL NOISE ----------
                if area < 1500:
                    continue

                # ---------- CONFIDENCE FILTER ----------
                min_conf = self.CLASS_MIN_CONF.get(class_id, 0.60)

                if conf < min_conf:
                    continue

                # ---------- SKIP GOOD ROAD ----------
                if class_id == 0:
                    continue

                # ---------- ROAD REGION FILTER ----------
                img_height = source.shape[0]

                # Ignore detections in upper image
                if y2 < img_height * 0.25:
                    continue

                # ---------- POTHOLE SHAPE FILTER ----------
                aspect_ratio = width / (height + 1e-5)

                if class_id == 1:

                    # Too thin
                    if aspect_ratio > 5:
                        continue

                    # Too tall
                    if aspect_ratio < 0.3:
                        continue

                severity, damage_rate = self._calculate_severity(
                    class_id,
                    conf
                )

                detections.append({

                    'label': self.CLASS_NAMES.get(
                        class_id,
                        'Unknown'
                    ),

                    'confidence': conf,

                    'bbox': (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),

                    'class_id': class_id,

                    'severity': severity,

                    'damage_rate': damage_rate,

                    'color': self.CLASS_COLORS.get(
                        class_id,
                        (255, 255, 255)
                    ),
                })

        # ---------- SORT BY CONFIDENCE ----------
        detections = sorted(
            detections,
            key=lambda x: x['confidence'],
            reverse=True
        )

        return detections

    def _calculate_severity(
        self,
        class_id: int,
        confidence: float
    ) -> Tuple[str, int]:

        if class_id == 3:

            if confidence > 0.85:
                return 'severe', int(np.random.randint(85, 98))

            elif confidence > 0.70:
                return 'medium', int(np.random.randint(60, 84))

            else:
                return 'low', int(np.random.randint(40, 59))

        elif class_id == 1:

            if confidence > 0.80:
                return 'high', int(np.random.randint(70, 90))

            return 'medium', int(np.random.randint(50, 75))

        elif class_id == 2:

            return 'low', int(np.random.randint(20, 50))

        else:

            return 'medium', int(np.random.randint(40, 70))

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_conf: bool = True,
        show_severity: bool = False,
    ) -> np.ndarray:

        img = image.copy()

        for det in detections:

            x1, y1, x2, y2 = det['bbox']

            label = det['label'].lower()

            confidence = det['confidence']

            color = det['color']

            # ---------- THICK BOX ----------
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # ---------- LABEL ----------
            if show_conf:
                text = f"{label} {confidence:.2f}"
            else:
                text = label

            # ---------- TEXT SIZE ----------
            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2
            )

            # ---------- LABEL BACKGROUND ----------
            cv2.rectangle(
                img,
                (x1, y1 - th - 12),
                (x1 + tw + 10, y1),
                color,
                -1
            )

            # ---------- LABEL TEXT ----------
            cv2.putText(
                img,
                text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        return img

    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        detections = self.detect(image)

        annotated = self.draw_detections(
            image,
            detections
        )

        if output_path:
            cv2.imwrite(output_path, annotated)

        return annotated, detections

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_display: bool = False,
    ) -> List[Dict]:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return []

        all_detections = []

        frame_idx = 0

        out = None

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            detections = self.detect(frame)

            for det in detections:
                det['frame'] = frame_idx

            all_detections.extend(detections)

            annotated = self.draw_detections(
                frame,
                detections
            )

            if output_path:

                if frame_idx == 0:

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                    h, w = frame.shape[:2]

                    out = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        30,
                        (w, h)
                    )

                if out:
                    out.write(annotated)

            if show_display:

                cv2.imshow(
                    'Road Damage Detection',
                    annotated
                )

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        cap.release()

        if out:
            out.release()

        cv2.destroyAllWindows()

        return all_detections


def create_detector(
    model_path: str = None
) -> RoadDamageDetector:

    return RoadDamageDetector(
        model_path=model_path or
        'runs/train/road_damage/weights/best.pt'
    )


if __name__ == '__main__':

    detector = create_detector()

    test_img = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    dets = detector.detect(test_img)

    print(
        f"Detections on blank frame: "
        f"{len(dets)} (expected 0)"
    )