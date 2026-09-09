"""
Test Script: Verify Good Road vs Damaged Road Classification
Run this after training to verify the fix
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


def create_test_images():
    """Create test images for verification"""
    test_dir = Path('data/test_images')
    test_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Clean good road
    img1 = np.zeros((640, 640, 3), dtype=np.uint8)
    img1[:] = (95, 95, 95)
    for y in range(0, 640, 50):
        cv2.rectangle(img1, (310, y), (330, y + 25), (230, 230, 230), -1)
    cv2.rectangle(img1, (40, 0), (55, 640), (0, 170, 230), -1)
    cv2.rectangle(img1, (585, 0), (600, 640), (0, 170, 230), -1)
    cv2.imwrite(str(test_dir / 'test_good_road.jpg'), img1)

    # Test 2: Pothole
    img2 = np.zeros((640, 640, 3), dtype=np.uint8)
    img2[:] = (90, 90, 90)
    for y in range(0, 640, 50):
        cv2.rectangle(img2, (310, y), (330, y + 25), (230, 230, 230), -1)
    cv2.circle(img2, (320, 400), 70, (20, 20, 20), -1)
    cv2.imwrite(str(test_dir / 'test_pothole.jpg'), img2)

    # Test 3: Accident
    img3 = np.zeros((640, 640, 3), dtype=np.uint8)
    img3[:] = (80, 80, 80)
    cv2.rectangle(img3, (200, 250), (440, 380), (50, 50, 50), -1)
    cv2.circle(img3, (250, 320), 20, (100, 100, 150), -1)
    cv2.circle(img3, (390, 320), 20, (100, 100, 150), -1)
    cv2.imwrite(str(test_dir / 'test_accident.jpg'), img3)

    print(f"Test images created in {test_dir}")
    return test_dir


def test_detection():
    """Test the trained model"""
    try:
        from ultralytics import YOLO

        model_path = 'runs/train/road_damage/weights/best.pt'

        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            print("Training first...")
            import subprocess
            subprocess.run([sys.executable, 'fix_training.py'])
            return

        print(f"Loading model: {model_path}")
        model = YOLO(model_path)

        test_dir = create_test_images()

        print("\n" + "="*60)
        print("TESTING GOOD ROAD vs DAMAGED ROAD CLASSIFICATION")
        print("="*60)

        # Test each image
        test_images = [
            ('test_good_road.jpg', 'Good Road'),
            ('test_pothole.jpg', 'Pothole'),
            ('test_accident.jpg', 'Accident')
        ]

        results_summary = []

        for img_file, expected in test_images:
            img_path = test_dir / img_file
            results = model(img_path, verbose=False)

            print(f"\n{img_file} (expected: {expected})")
            print("-" * 40)

            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    detections.append((cls_id, conf))

            if not detections:
                print("  ✓ No detections (clean road)")
                results_summary.append(('good', expected == 'Good Road'))
            else:
                for cls_id, conf in detections:
                    class_names = {0: 'Good Road', 1: 'Pothole', 2: 'Crack', 3: 'Accident'}
                    print(f"  Detected: {class_names.get(cls_id, 'Unknown')} ({conf:.2f})")
                    results_summary.append((class_names.get(cls_id, 'Unknown'), expected))

        # Summary
        print("\n" + "="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)

        correct = 0
        for detected, expected in results_summary:
            status = "✓" if detected == expected else "✗"
            print(f"  {status} Expected: {expected}, Got: {detected}")
            if detected == expected or (expected == 'Good Road' and detected == 'Good Road'):
                correct += 1

        accuracy = correct / len(results_summary) * 100
        print(f"\nAccuracy: {accuracy:.1f}%")

        if accuracy >= 70:
            print("\n✓ Model is correctly classifying good vs damaged roads!")
        else:
            print("\n✗ Model needs more training. Try more epochs.")

    except ImportError:
        print("Install ultralytics: pip install ultralytics")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    test_detection()