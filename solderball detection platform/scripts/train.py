"""
Training Script for Solder Ball Detection using Ultralytics YOLO.

Usage:
    python scripts/train.py --epochs 50 --imgsz 640 --batch 4
"""

import os
import shutil
import argparse
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on annotated solder ball dataset.")
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pretrained base model (e.g. yolov8n.pt, yolo11n.pt)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (640 for fast CPU, 1080/1280 for tiny solder balls)')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--workers', type=int, default=2, help='Dataloader workers')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    args = parser.parse_args()

    # Check if data exists
    if not os.path.exists(args.data):
        print(f"Error: YAML config '{args.data}' not found!")
        return

    train_imgs = os.listdir('data/images/train') if os.path.exists('data/images/train') else []
    if not train_imgs:
        print("Error: No training images found in 'data/images/train'!")
        print("Please annotate images and run 'python scripts/split_dataset.py' first.")
        return

    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} ({'GPU' if device == '0' else 'CPU'})")
    print(f"Dataset: {len(train_imgs)} training images")
    print(f"Starting training with base model '{args.model}' for {args.epochs} epochs...")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        patience=args.patience,
        name='solderball_run'
    )

    # Copy best model to models/
    best_weights = os.path.join(results.save_dir, 'weights', 'best.pt')
    os.makedirs('models', exist_ok=True)
    dst_weights = os.path.join('models', 'best_solderball_detector.pt')
    if os.path.exists(best_weights):
        shutil.copy2(best_weights, dst_weights)
        print(f"\nTraining completed successfully!")
        print(f"Best weights saved to: {dst_weights}")
    else:
        print(f"\nTraining completed. Weights located in: {results.save_dir}")

if __name__ == '__main__':
    main()
