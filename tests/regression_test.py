import argparse
import json
import os
import time
import pandas as pd
from glob import glob
import sys
import cv2
from tqdm import tqdm

# Добавляем путь к src в sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.yolov26n_detector import YOLOV26NPersonDetector
from models.yolov11n_detector import YOLOV11NPersonDetector

def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    annotations = {}
    for _, row in df.iterrows():
        filename = row['filename']
        if filename not in annotations:
            annotations[filename] = []
        annotations[filename].append({
            'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
            'class': row['class']
        })
    return annotations

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def evaluate_predictions(predictions, annotations, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    
    for pred in predictions:
        matched = False
        for ann in annotations:
            if calculate_iou(pred['bbox'], ann['bbox']) >= iou_threshold:
                matched = True
                break
        if matched:
            tp += 1
        else:
            fp += 1
    
    fn = len(annotations) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

def test_model(model_class, data_dir, annotations):
    detector = model_class()
    
    image_files = glob(os.path.join(data_dir, "*.jpg")) + glob(os.path.join(data_dir, "*.png"))
    
    results = []
    total_time = 0
    
    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        
        start_time = time.time()
        predictions = detector.detect(image)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        img_annotations = annotations.get(filename, [])
        metrics = evaluate_predictions(predictions, img_annotations)
        
        results.append({
            "image": filename,
            "detections": predictions,
            "count": len(predictions),
            "time": elapsed,
            "metrics": metrics
        })
    
    avg_time = total_time / len(image_files) if image_files else 0
    
    return results, avg_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, 
                       choices=['yolov26n', 'yolov11n'],
                       help="Model names to test")
    parser.add_argument("--data_dir", default="data/regression_data", 
                       help="Test data directory")
    parser.add_argument("--annotations", default="data/regression_data/annotations.csv", 
                       help="Annotations CSV file")
    args = parser.parse_args()
    
    annotations = load_annotations(args.annotations)
    
    model_classes = {
        'yolov26n': YOLOV26NPersonDetector,
        'yolov11n': YOLOV11NPersonDetector
    }
    
    all_results = {}
    summary = []
    
    for model_name in args.models:
        if model_name not in model_classes:
            print(f"Unknown model: {model_name}")
            continue
            
        print(f"\nTesting {model_name}")
        results, avg_time = test_model(model_classes[model_name], args.data_dir, annotations)
        all_results[model_name] = results
        
        total_detections = sum(r["count"] for r in results)
        total_time = sum(r["time"] for r in results)
        
        avg_precision = sum(r["metrics"]["precision"] for r in results) / len(results) if results else 0
        avg_recall = sum(r["metrics"]["recall"] for r in results) / len(results) if results else 0
        avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results) if results else 0
        
        print(f"Total images: {len(results)}")
        print(f"Total detections: {total_detections}")
        print(f"Average precision: {avg_precision:.3f}")
        print(f"Average recall: {avg_recall:.3f}")
        print(f"Average F1: {avg_f1:.3f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per image: {avg_time:.3f}s")
        
        summary.append({
            'model': model_name,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'avg_time': avg_time,
            'total_detections': total_detections
        })
    
    with open("regression_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*50)
    print("SUMMARY:")
    for s in summary:
        print(f"{s['model']}: F1={s['avg_f1']:.3f}, Time={s['avg_time']:.3f}s")

if __name__ == "__main__":
    main()