import argparse
import json
import os
import time
import pandas as pd
from glob import glob
import sys
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.yolov26n_detector import YOLOV26NPersonDetector
from models.yolov11n_detector import YOLOV11NPersonDetector

def load_annotations(csv_path):
    """Загружает аннотации bounding box из CSV файла."""
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
    """Вычисляет Intersection over Union (IoU) для двух bounding boxes."""
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

def evaluate_at_threshold(predictions, annotations, conf_threshold, iou_threshold=0.5):
    """Вычисляет precision, recall и F1 для заданного порога уверенности."""
    tp, fp, fn = 0, 0, 0
    
    filtered_preds = [p for p in predictions if p.get('confidence', 1.0) >= conf_threshold]
    
    for pred in filtered_preds:
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

def calculate_average_precision(precisions, recalls):
    """Вычисляет Average Precision (AP) как площадь под кривой Precision-Recall[citation:2][citation:6]."""
    # Аппроксимация по методу COCO: интерполируем precision для каждого уровня recall
    recalls_interp = np.linspace(0, 1, 101)  # 101 точка
    precisions_interp = np.zeros_like(recalls_interp)
    for i, r in enumerate(recalls_interp):
        precisions_at_recall = precisions[recalls >= r]
        precisions_interp[i] = precisions_at_recall.max() if len(precisions_at_recall) > 0 else 0
    ap = precisions_interp.mean()  # Площадь под интерполированной кривой
    return ap

def calculate_map(results, confidence_thresholds, iou_threshold=0.5):
    """Вычисляет mAP (Mean Average Precision) для заданного порога IoU."""
    # Для одного класса 'person' mAP = AP
    all_precisions = []
    all_recalls = []
    
    for conf_thresh in confidence_thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for img_result in results:
            predictions = img_result['detections']
            annotations = img_result['annotations']
            
            img_tp, img_fp = 0, 0
            matched_annotations = [False] * len(annotations)
            
            filtered_preds = [p for p in predictions if p.get('confidence', 1.0) >= conf_thresh]
            filtered_preds_sorted = sorted(filtered_preds, key=lambda x: x.get('confidence', 1.0), reverse=True)
            
            for pred in filtered_preds_sorted:
                best_iou = 0
                best_idx = -1
                
                for i, ann in enumerate(annotations):
                    if matched_annotations[i]:
                        continue
                    
                    iou = calculate_iou(pred['bbox'], ann['bbox'])
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                if best_idx >= 0:
                    img_tp += 1
                    matched_annotations[best_idx] = True
                else:
                    img_fp += 1
            
            img_fn = sum(1 for matched in matched_annotations if not matched)
            
            total_tp += img_tp
            total_fp += img_fp
            total_fn += img_fn
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
    
    # Вычисляем AP (для одного класса это и есть mAP)
    ap = calculate_average_precision(np.array(all_precisions), np.array(all_recalls))
    return ap

def calculate_map_range(results, confidence_thresholds):
    """Вычисляет mAP@0.5:0.95 (среднее по порогам IoU от 0.5 до 0.95 с шагом 0.05)."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    
    for iou_thresh in iou_thresholds:
        ap = calculate_map(results, confidence_thresholds, iou_threshold=iou_thresh)
        aps.append(ap)
    
    # Среднее значение AP по всем порогам IoU[citation:6][citation:8]
    map_50_95 = np.mean(aps) if aps else 0
    return map_50_95

def test_model(model_class, data_dir, annotations):
    """Запускает модель на тестовых данных и возвращает результаты."""
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
        
        results.append({
            "image": filename,
            "detections": predictions,
            "annotations": img_annotations,
            "time": elapsed
        })
    
    avg_time = total_time / len(image_files) if image_files else 0
    
    return results, avg_time

def calculate_metrics_vs_confidence(results, confidence_thresholds):
    """Вычисляет mAP@50 и mAP@50-95 для различных порогов уверенности."""
    map50_scores = []
    map50_95_scores = []
    
    for conf_thresh in confidence_thresholds:
        # Отфильтровываем результаты по порогу уверенности
        filtered_results = []
        for img_result in results:
            filtered_dets = [det for det in img_result['detections'] if det.get('confidence', 1.0) >= conf_thresh]
            if filtered_dets:
                filtered_results.append({
                    'image': img_result['image'],
                    'detections': filtered_dets,
                    'annotations': img_result['annotations'],
                    'time': img_result['time']
                })
            else:
                filtered_results.append({
                    'image': img_result['image'],
                    'detections': [],
                    'annotations': img_result['annotations'],
                    'time': img_result['time']
                })
        
        # Вычисляем mAP@50 для текущего порога уверенности
        map50 = calculate_map(filtered_results, [conf_thresh], iou_threshold=0.5)
        map50_scores.append(map50)
        
        # Вычисляем mAP@50-95 для текущего порога уверенности
        map50_95 = calculate_map_range(filtered_results, [conf_thresh])
        map50_95_scores.append(map50_95)
    
    return map50_scores, map50_95_scores

def plot_map_vs_confidence(all_results, confidence_thresholds):
    """Строит графики mAP@50 и mAP@50-95 в зависимости от порога уверенности."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for model_name, results in all_results.items():
        map50_scores, map50_95_scores = calculate_metrics_vs_confidence(results, confidence_thresholds)
        
        ax1.plot(confidence_thresholds, map50_scores, marker='o', label=model_name, linewidth=2)
        ax2.plot(confidence_thresholds, map50_95_scores, marker='s', label=model_name, linewidth=2)
    
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('mAP@0.5')
    ax1.set_title('mAP@0.5 vs Confidence Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('mAP@0.5:0.95')
    ax2.set_title('mAP@0.5:0.95 vs Confidence Threshold[citation:8]')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('map_vs_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    
    confidence_thresholds = np.linspace(0.01, 0.9, 90)  # Более детальный шаг
    
    for model_name in args.models:
        if model_name not in model_classes:
            print(f"Unknown model: {model_name}")
            continue
            
        print(f"\nTesting {model_name}")
        results, avg_time = test_model(model_classes[model_name], args.data_dir, annotations)
        all_results[model_name] = results
        
        # Вычисляем mAP@50 и mAP@50-95 для порога уверенности 0.5
        print(f"Calculating mAP metrics...")
        map50 = calculate_map(results, [0.5], iou_threshold=0.5)
        map50_95 = calculate_map_range(results, [0.5])
        
        # Старые метрики для полноты (можно убрать)
        metrics_at_05 = evaluate_at_threshold(
            [det for r in results for det in r['detections']],
            [ann for r in results for ann in r['annotations']],
            0.5
        )
        
        total_detections = sum(len(r["detections"]) for r in results)
        total_time = sum(r["time"] for r in results)
        
        print(f"Total images: {len(results)}")
        print(f"Total detections: {total_detections}")
        print(f"mAP@0.5: {map50:.4f}")
        print(f"mAP@0.5:0.95: {map50_95:.4f}")
        print(f"Precision @0.5: {metrics_at_05['precision']:.3f}")
        print(f"Recall @0.5: {metrics_at_05['recall']:.3f}")
        print(f"F1 @0.5: {metrics_at_05['f1']:.3f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per image: {avg_time:.3f}s")
        
        summary.append({
            'model': model_name,
            'map50': map50,
            'map50_95': map50_95,
            'precision': metrics_at_05['precision'],
            'recall': metrics_at_05['recall'],
            'f1': metrics_at_05['f1'],
            'avg_time': avg_time,
            'total_detections': total_detections
        })
    
    # Сохраняем результаты с новыми метриками
    output_data = {}
    for model_name, results in all_results.items():
        model_summary = [s for s in summary if s['model'] == model_name][0]
        output_data[model_name] = {
            'summary': model_summary,
            'results': [{'image': r['image'], 'detections': r['detections'], 'time': r['time']} for r in results]
        }
    
    with open("regression_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*50)
    print("SUMMARY:")
    for s in summary:
        print(f"{s['model']}: mAP@50={s['map50']:.4f}, mAP@50-95={s['map50_95']:.4f}, Time={s['avg_time']:.3f}s")
    
    plot_map_vs_confidence(all_results, confidence_thresholds)
    print("\nGraph saved: map_vs_confidence.png")

if __name__ == "__main__":
    main()