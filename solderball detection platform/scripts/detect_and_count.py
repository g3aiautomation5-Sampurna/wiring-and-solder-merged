"""
Solder Ball Detection and Counting Script.
Detects all solder balls in an image or directory, counts them, and generates annotated outputs.

Usage:
    python scripts/detect_and_count.py --source dataset/solderball\ (1).jpg --conf 0.25
    python scripts/detect_and_count.py --source dataset/ --limit 10
"""

import os
import glob
import csv
import argparse
import cv2
import numpy as np

def run_yolo_detection(model, image, conf_thresh=0.25, iou_thresh=0.45):
    """Runs YOLO model inference and returns detected boxes."""
    results = model.predict(image, conf=conf_thresh, iou=iou_thresh, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            detections.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'conf': conf,
                'class_id': cls_id
            })
    return detections

def run_fallback_cv_detection(image):
    """
    Fallback Computer Vision detector (Hough Circles + Color filtering)
    Used when a trained YOLO model is not yet available.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circular solder balls
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=15, 
        param1=50, 
        param2=28, 
        minRadius=3, 
        maxRadius=25
    )
    
    detections = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            x, y, r = c[0], c[1], c[2]
            detections.append({
                'box': (int(x - r), int(y - r), int(x + r), int(y + r)),
                'conf': 0.85,
                'class_id': 0
            })
    return detections

def draw_detections_and_count(image, detections, image_name=""):
    """Draws bounding boxes, index tags, and total count banner."""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    total_count = len(detections)

    # Draw boxes
    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['box']
        conf = det['conf']
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # High-visibility neon bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Center marker dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)

        # Index and confidence label
        label = f"#{idx} ({conf:.2f})"
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        lbl_y1 = max(0, y1 - th - 6)
        lbl_y2 = y1
        cv2.rectangle(annotated, (x1, lbl_y1), (x1 + tw + 6, lbl_y2), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Top overlay header banner
    banner_height = 50
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Banner text: Count only
    status_text = f"SOLDER BALLS FOUND: {total_count}"
    status_color = (0, 255, 120)
    
    cv2.putText(annotated, status_text, (20, 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, status_color, 2)
    
    if image_name:
        cv2.putText(annotated, f"File: {image_name}", (w - 400, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    return annotated, total_count

def main():
    parser = argparse.ArgumentParser(description="Detect and count solder balls.")
    parser.add_argument('--source', type=str, default='dataset', help='Image path or directory of images')
    parser.add_argument('--model', type=str, default='models/best_solderball_detector.pt', help='Trained YOLO model path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU NMS threshold')
    parser.add_argument('--output-dir', type=str, default='detection_results', help='Output directory')
    parser.add_argument('--limit', type=int, default=10, help='Limit processing when source is a directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load YOLO model or fall back to OpenCV detector
    yolo_model = None
    if os.path.exists(args.model):
        from ultralytics import YOLO
        print(f"Loading trained model: {args.model}")
        yolo_model = YOLO(args.model)
    else:
        print(f"Note: Trained model '{args.model}' not found.")
        print("Using Computer Vision fallback detector until model training completes.")

    # Collect source images
    if os.path.isfile(args.source):
        image_files = [args.source]
    elif os.path.isdir(args.source):
        image_files = glob.glob(os.path.join(args.source, '*.jpg')) + \
                      glob.glob(os.path.join(args.source, '*.png')) + \
                      glob.glob(os.path.join(args.source, '*.jpeg'))
        image_files = sorted(image_files)[:args.limit]
    else:
        print(f"Error: Invalid source path '{args.source}'")
        return

    csv_log_path = os.path.join(args.output_dir, 'detection_summary.csv')
    csv_rows = []

    print(f"\nProcessing {len(image_files)} image(s)...")
    print("-" * 65)

    for img_p in image_files:
        base_name = os.path.basename(img_p)
        img = cv2.imread(img_p)
        if img is None:
            continue

        if yolo_model:
            detections = run_yolo_detection(yolo_model, img, args.conf, args.iou)
        else:
            detections = run_fallback_cv_detection(img)

        annotated_img, count = draw_detections_and_count(img, detections, base_name)

        out_img_path = os.path.join(args.output_dir, f"detected_{base_name}")
        cv2.imwrite(out_img_path, annotated_img)

        csv_rows.append({
            'filename': base_name,
            'solder_ball_count': count,
            'detections': [(d['box'], round(d['conf'], 3)) for d in detections]
        })

        print(f"[{base_name}] -> Found {count} solder ball(s) | Saved to {out_img_path}")

    # Write summary CSV
    with open(csv_log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'solder_ball_count', 'detections'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("-" * 65)
    print(f"Batch detection complete!")
    print(f"Summary log written to: {csv_log_path}")
    print(f"Annotated images saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()
