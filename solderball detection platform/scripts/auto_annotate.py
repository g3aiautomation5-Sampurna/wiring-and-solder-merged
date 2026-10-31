"""
Auto-Annotation / Active Learning Script.
Uses a trained or fine-tuned model to predict bounding boxes on unannotated images
and automatically generates YOLO format label files.

Usage:
    python scripts/auto_annotate.py --model models/best_solderball_detector.pt --conf 0.3
"""

import os
import glob
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Auto-annotate images using trained model.")
    parser.add_argument('--dataset-dir', type=str, default='dataset', help='Directory with images to annotate')
    parser.add_argument('--ann-dir', type=str, default='annotations', help='Directory to save output .txt labels')
    parser.add_argument('--model', type=str, default='models/best_solderball_detector.pt', help='Trained model weights')
    parser.add_argument('--conf', type=float, default=0.30, help='Confidence threshold for auto-annotation')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing annotations')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train a model first using 'python scripts/train.py'")
        return

    os.makedirs(args.ann_dir, exist_ok=True)
    model = YOLO(args.model)

    image_files = glob.glob(os.path.join(args.dataset_dir, '*.jpg')) + \
                  glob.glob(os.path.join(args.dataset_dir, '*.png'))

    annotated_count = 0
    skipped_count = 0

    print(f"Running auto-annotation with '{args.model}' (conf >= {args.conf})...")

    for img_p in image_files:
        base = os.path.splitext(os.path.basename(img_p))[0]
        out_txt = os.path.join(args.ann_dir, f"{base}.txt")

        if os.path.exists(out_txt) and not args.overwrite:
            skipped_count += 1
            continue

        results = model.predict(img_p, conf=args.conf, verbose=False)
        lines = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # YOLO format: cls, x_center, y_center, width, height (normalized)
                xywhn = box.xywhn[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                lines.append(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")

        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        annotated_count += 1
        print(f"Generated {len(lines)} labels for {base} -> {out_txt}")

    print("\nAuto-annotation completed:")
    print(f"  Newly annotated : {annotated_count} images")
    print(f"  Skipped (exist) : {skipped_count} images")
    print(f"  Labels saved to : {args.ann_dir}/")
    print("Tip: Review boxes using 'python scripts/verify_annotations.py'!")

if __name__ == '__main__':
    main()
