"""
Verification Script for Annotations.
Renders bounding boxes and ball counts onto sample images to verify that:
1. Coordinates are accurate and aligned.
2. Solder balls are properly boxed.
3. Box sizes and aspects are reasonable.

Usage:
    python scripts/verify_annotations.py --limit 10
"""

import os
import glob
import cv2
import argparse

def parse_yolo_label(lbl_path, img_w, img_h):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)
                boxes.append((cls_id, x1, y1, x2, y2))
    return boxes

def main():
    parser = argparse.ArgumentParser(description="Verify annotated solder ball boxes.")
    parser.add_argument('--source-dir', type=str, default='data', help="Directory containing images and labels (or 'dataset')")
    parser.add_argument('--output-dir', type=str, default='verification_output', help="Where to save verified images")
    parser.add_argument('--limit', type=int, default=15, help="Number of annotated images to inspect")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Search in data/images or dataset
    if os.path.exists(os.path.join(args.source_dir, 'images')):
        img_pattern = os.path.join(args.source_dir, 'images', '**', '*.jpg')
        img_files = glob.glob(img_pattern, recursive=True)
    else:
        img_files = glob.glob(os.path.join(args.source_dir, '*.jpg'))

    inspected = 0
    total_balls = 0

    for img_p in img_files:
        base = os.path.splitext(os.path.basename(img_p))[0]
        # Find label file
        parent = os.path.dirname(img_p)
        possible_label_paths = [
            img_p.replace('images', 'labels').replace('.jpg', '.txt'),
            os.path.join(parent, f"{base}.txt"),
            os.path.join('annotations', f"{base}.txt"),
            os.path.join('dataset', f"{base}.txt")
        ]
        
        lbl_file = None
        for p in possible_label_paths:
            if os.path.exists(p):
                lbl_file = p
                break
        
        if not lbl_file:
            continue

        img = cv2.imread(img_p)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        boxes = parse_yolo_label(lbl_file, iw, ih)
        if not boxes:
            continue

        count = len(boxes)
        total_balls += count

        # Draw boxes and counts
        canvas = img.copy()
        for i, (cls_id, x1, y1, x2, y2) in enumerate(boxes, 1):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(canvas, f"#{i}", (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Draw top banner with count
        cv2.rectangle(canvas, (0, 0), (iw, 45), (0, 0, 0), -1)
        cv2.putText(canvas, f"Image: {os.path.basename(img_p)} | Solder Balls: {count}", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out_path = os.path.join(args.output_dir, f"verified_{base}.jpg")
        cv2.imwrite(out_path, canvas)
        inspected += 1

        print(f"Verified {base}: found {count} solder ball(s) -> saved to {out_path}")
        if inspected >= args.limit:
            break

    print(f"\nVerification summary:")
    print(f"  Inspected images : {inspected}")
    print(f"  Total balls seen : {total_balls}")
    print(f"  Outputs saved in : {args.output_dir}/")

if __name__ == '__main__':
    main()
