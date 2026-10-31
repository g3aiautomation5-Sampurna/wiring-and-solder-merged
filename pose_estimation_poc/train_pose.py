"""
Training Script for Wire Harness Keypoint Estimation POC (YOLOv8-Pose)
Uses Ultralytics to train on wire keypoint data.
"""

import os
import sys
from ultralytics import YOLO
import torch

def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(script_dir, "data.yaml")

    # Check that dataset has labeled files
    train_labels = os.path.join(script_dir, "dataset", "labels", "train")
    if not os.path.exists(train_labels) or len(os.listdir(train_labels)) == 0:
        print(f"\n[ERROR] No training labels found in: {train_labels}")
        print("Please annotate at least 10-20 images first by running:")
        print("    python annotate_wires.py")
        sys.exit(1)

    # Clear existing YOLO label cache to ensure newly annotated samples are picked up
    import glob
    for c_file in glob.glob(os.path.join(script_dir, "dataset", "labels", "*.cache")):
        try:
            os.remove(c_file)
        except Exception:
            pass

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Starting YOLOv8-Pose Training")
    print(f"Data config: {data_yaml}")
    print(f"Device: {device}")
    print(f"=======================================================\n")

    # Load pre-trained nano pose model for fast training and low resource usage
    model = YOLO("yolov8n-pose.pt")

    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        fliplr=0.0,
        project=os.path.join(script_dir, "runs"),
        name="wire_pose_model",
        exist_ok=True,
        plots=True
    )

    print("\nTraining completed successfully!")
    print(f"Best model weights saved to: {os.path.join(script_dir, 'runs', 'wire_pose_model', 'weights', 'best.pt')}")

if __name__ == "__main__":
    train()
