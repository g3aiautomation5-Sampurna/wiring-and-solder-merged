"""
Train YOLOv8 on Origin 3 Dataset
Uses Ultralytics YOLOv8 nano model (fast & accurate for real-time detection).
"""

import os
import torch
from ultralytics import YOLO

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(WORKSPACE_DIR, "yolo_dataset", "data.yaml")


def main():
    if not os.path.exists(YAML_PATH):
        print(f"[ERROR] {YAML_PATH} not found!")
        print("Please run `python prepare_dataset.py` first after finishing annotation.")
        return

    # Check hardware
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"--> Using device: {device} ({'GPU' if device == '0' else 'CPU'})")

    # Load pre-trained nano model
    print("--> Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    # Train
    print("--> Starting training...")
    results = model.train(
        data=YAML_PATH,
        epochs=40,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        plots=True,
        save=True,
        project=os.path.join(WORKSPACE_DIR, "runs"),
        name="origin3_model",
        exist_ok=True,
    )

    best_weights = os.path.join(WORKSPACE_DIR, "runs", "origin3_model", "weights", "best.pt")
    print("\n==================================================")
    print("Training Complete!")
    print(f"Best trained weights: {best_weights}")
    print("==================================================")
    print("You can now run inference with:\n  python detect.py")


if __name__ == "__main__":
    main()
