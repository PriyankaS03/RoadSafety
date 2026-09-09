"""
utils/detection.py – AI detection engine with real YOLOv8 + webcam support
damage_rate is generated per detection and passed to add_report.
"""
import os
import sys
import cv2
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from utils.db import add_report

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent / 'backend'
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Try to import YOLOv8
try:
    from backend.detect import RoadDamageDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available, using mock detection")

TN_LOCATIONS = [
    {"name": "Gandhipuram, Coimbatore",        "lat": 11.0168, "lon": 76.9558},
    {"name": "RS Puram, Coimbatore",            "lat": 10.9858, "lon": 76.9551},
    {"name": "Ukkadam, Coimbatore",             "lat": 10.9942, "lon": 76.9629},
    {"name": "Peelamedu, Coimbatore",           "lat": 11.0235, "lon": 77.0229},
    {"name": "Narasipuram, Coimbatore",         "lat": 11.0280, "lon": 76.9630},
    {"name": "Saibaba Colony, Coimbatore",      "lat": 11.0190, "lon": 76.9401},
    {"name": "Singanallur, Coimbatore",         "lat": 11.0026, "lon": 77.0254},
    {"name": "T. Nagar, Chennai",               "lat": 13.0404, "lon": 80.2331},
    {"name": "Adyar, Chennai",                  "lat": 13.0012, "lon": 80.2565},
    {"name": "Anna Nagar, Chennai",             "lat": 13.0850, "lon": 80.2101},
    {"name": "Velachery, Chennai",              "lat": 12.9815, "lon": 80.2180},
    {"name": "Tambaram, Chennai",               "lat": 12.9249, "lon": 80.1000},
    {"name": "Meenakshi Amman Kovil, Madurai",  "lat": 9.9195,  "lon": 78.1191},
    {"name": "Tallakulam, Madurai",             "lat": 9.9307,  "lon": 78.1370},
    {"name": "Srirangam, Trichy",               "lat": 10.8660, "lon": 78.6905},
    {"name": "Thillai Nagar, Trichy",            "lat": 10.8077, "lon": 78.6920},
    {"name": "Fairlands, Salem",                "lat": 11.6643, "lon": 78.1460},
    {"name": "Shevapet, Salem",                 "lat": 11.6595, "lon": 78.1540},
    {"name": "Erode Bus Stand",                 "lat": 11.3410, "lon": 77.7172},
    {"name": "Palayamkottai, Tirunelveli",      "lat": 8.7139,  "lon": 77.7567},
    {"name": "Katpadi, Vellore",                "lat": 12.9563, "lon": 79.1397},
    {"name": "Charring Cross, Ooty",            "lat": 11.4102, "lon": 76.6950},
]

POTHOLE_SEVERITIES = ["low", "low", "medium", "medium", "medium", "severe"]

# Global detector instance
_detector = None


def get_detector() -> Optional[RoadDamageDetector]:
    """Get or create the YOLOv8 detector instance"""
    global _detector

    if _detector is None and YOLO_AVAILABLE:
        try:
            model_path = 'runs/train/road_damage/weights/best.pt'
            if Path(model_path).exists():
                _detector = RoadDamageDetector(model_path=model_path, conf_threshold=0.3)
            else:
                print(f"Model not found at {model_path}, using default YOLOv8")
                _detector = RoadDamageDetector(conf_threshold=0.3)
        except Exception as e:
            print(f"Error loading detector: {e}")
            _detector = None

    return _detector


def simulate_event(event_type: str):
    """Called from sidebar demo buttons. Generates realistic damage_rate."""
    loc = random.choice(TN_LOCATIONS)
    if event_type == "accident":
        severity    = random.choice(["severe", "severe", "medium", "low"])
        damage_rate = (
            random.randint(80, 98) if severity == "severe" else
            random.randint(50, 79) if severity == "medium" else
            random.randint(5,  49)
        )
        add_report("accident", severity, loc["name"], loc["lat"], loc["lon"],
                   damage_rate=damage_rate)
    elif event_type == "pothole":
        sev = random.choice(POTHOLE_SEVERITIES)
        add_report("pothole", sev, loc["name"], loc["lat"], loc["lon"])


def classify_pothole_severity(count: int) -> str:
    if count <= 2:   return "low"
    elif count <= 5: return "medium"
    return "severe"


def mock_frame_detection(frame=None):
    """
    Mock detection for fallback when YOLO not available.
    Returns list of detection dicts including damage_rate.
    """
    detections = []
    r = random.random()
    if r < 0.15:
        severity    = "severe"
        damage_rate = random.randint(80, 98)
        detections.append({
            "label":       "Accident",
            "confidence":  round(random.uniform(0.72, 0.96), 2),
            "bbox":        (50, 40, 320, 230),
            "severity":    severity,
            "damage_rate": damage_rate,
            "color":       (0, 0, 255),
        })
    elif r < 0.30:
        severity    = "medium"
        damage_rate = random.randint(50, 79)
        detections.append({
            "label":       "Accident",
            "confidence":  round(random.uniform(0.60, 0.85), 2),
            "bbox":        (60, 50, 280, 200),
            "severity":    severity,
            "damage_rate": damage_rate,
            "color":       (0, 100, 255),
        })
    elif r < 0.42:
        severity    = "low"
        damage_rate = random.randint(5, 49)
        detections.append({
            "label":       "Accident",
            "confidence":  round(random.uniform(0.50, 0.72), 2),
            "bbox":        (70, 60, 240, 180),
            "severity":    severity,
            "damage_rate": damage_rate,
            "color":       (0, 150, 255),
        })
    elif r < 0.70:
        n = random.randint(1, 6)
        for _ in range(n):
            sev = classify_pothole_severity(n)
            detections.append({
                "label":       "Pothole",
                "confidence":  round(random.uniform(0.65, 0.92), 2),
                "bbox":        (random.randint(10, 250), random.randint(100, 300),
                                random.randint(60, 160), random.randint(40, 100)),
                "severity":    sev,
                "damage_rate": 0,
                "color":       (0, 165, 255),
            })
    return detections


def real_frame_detection(frame) -> List[Dict]:
    """
    Real YOLOv8 inference on frame.
    Returns list of detection dicts including damage_rate.
    """
    detector = get_detector()

    if detector is None:
        return mock_frame_detection(frame)

    try:
        # Run detection
        results = detector.detect(frame)

        # Convert to expected format
        detections = []
        for det in results:
            detections.append({
                "label":       det.get('label', 'Unknown'),
                "confidence":  det.get('confidence', 0.0),
                "bbox":        det.get('bbox', (0, 0, 100, 100)),
                "severity":    det.get('severity', 'low'),
                "damage_rate": det.get('damage_rate', 0),
                "color":       det.get('color', (0, 165, 255)),
                "class_id":    det.get('class_id', 4)
            })

        return detections

    except Exception as e:
        print(f"Detection error: {e}")
        return mock_frame_detection(frame)


def is_accident(detections: List[Dict]) -> bool:
    """Check if any detection is an accident"""
    for det in detections:
        if det.get('label', '').lower() == 'accident':
            return True
    return False


def get_accident_severity(detections: List[Dict]) -> str:
    """Get the highest severity accident from detections"""
    max_severity = None
    max_damage = 0

    for det in detections:
        if det.get('label', '').lower() == 'accident':
            damage = det.get('damage_rate', 0)
            if damage > max_damage:
                max_damage = damage
                max_severity = det.get('severity', 'low')

    return max_severity or 'none'


def process_webcam(
    camera_index: int = 0,
    save_output: bool = False,
    output_path: str = 'output.mp4'
):
    """Process webcam stream with real-time detection"""
    detector = get_detector()

    if detector is None:
        print("Using mock detection for webcam")
        detector = type('obj', (object,), {'detect': mock_frame_detection})()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    writer = None
    if save_output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    print("Starting webcam detection... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        if YOLO_AVAILABLE and get_detector():
            detections = real_frame_detection(frame)
        else:
            detections = mock_frame_detection(frame)

        # Draw
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            label = det['label']
            conf = det['confidence']

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show
        cv2.imshow('Road Damage Detection', frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def get_random_tn_location():
    return random.choice(TN_LOCATIONS)


# Export main functions
__all__ = [
    'simulate_event',
    'classify_pothole_severity',
    'mock_frame_detection',
    'real_frame_detection',
    'is_accident',
    'get_accident_severity',
    'process_webcam',
    'get_random_tn_location',
    'get_detector',
    'TN_LOCATIONS'
]