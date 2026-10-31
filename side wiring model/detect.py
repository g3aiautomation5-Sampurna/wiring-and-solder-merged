"""
Detect Origin 3 using Trained YOLO Model
Runs inference on test images and displays the detected Origin 3 with bounding box and center coordinate.
"""

import os
import glob
import cv2
from ultralytics import YOLO

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKSPACE_DIR, "runs", "origin3_model", "weights", "best.pt")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "predictions")


def run_inference(image_path=None, conf_threshold=0.4):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model weights not found at: {MODEL_PATH}")
        print("Please train your model first by running `python train.py`.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    # If no specific image given, take sample images
    if image_path is None:
        targets = glob.glob(os.path.join(WORKSPACE_DIR, "yolo_dataset", "images", "val", "*.jpg"))
        if not targets:
            targets = glob.glob(os.path.join(WORKSPACE_DIR, "dataset", "*.jpg"))
        targets = targets[:5]
    elif os.path.isdir(image_path):
        targets = glob.glob(os.path.join(image_path, "*.jpg")) + glob.glob(os.path.join(image_path, "*.png"))
    else:
        targets = [image_path]

    for target in targets:
        print(f"\n--> Running detection on: {os.path.basename(target)}")
        results = model.predict(source=target, conf=conf_threshold, verbose=False)

        for result in results:
            boxes = result.boxes
            print(f"Detected {len(boxes)} object(s):")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                print(f"  [{i+1}] Class: {model.names[cls_id]} | Conf: {conf:.2f} | BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] | Center: ({cx:.1f}, {cy:.1f})")

            # Save annotated output image
            plotted_img = result.plot()
            out_filename = "pred_" + os.path.basename(target)
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            cv2.imwrite(out_path, plotted_img)
            print(f"--> Saved to: {out_path}")


if __name__ == "__main__":
    import sys
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_inference(img_arg)
