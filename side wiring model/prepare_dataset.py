"""
Prepare Dataset for YOLO Training
Validates annotations, splits dataset into Train (80%) and Val (20%),
and generates data.yaml ready for Ultralytics YOLO.
"""

import os
import glob
import shutil
import random
import yaml

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
YOLO_DIR = os.path.join(WORKSPACE_DIR, "yolo_dataset")
CLASS_NAME = "origin_3"


def prepare_yolo_dataset(split_ratio=0.8, seed=42):
    random.seed(seed)

    all_images = sorted(
        glob.glob(os.path.join(DATASET_DIR, "*.jpg")) +
        glob.glob(os.path.join(DATASET_DIR, "*.png")) +
        glob.glob(os.path.join(DATASET_DIR, "*.jpeg"))
    )

    if not all_images:
        print(f"[ERROR] No images found in {DATASET_DIR}")
        return False

    # Check annotations
    labeled_pairs = []
    unlabeled_images = []

    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            labeled_pairs.append((img_path, lbl_path))
        else:
            unlabeled_images.append(img_path)

    total_imgs = len(all_images)
    total_labeled = len(labeled_pairs)

    print(f"\n==================================================")
    print(f" Dataset Summary: {total_labeled}/{total_imgs} images labeled ({int(total_labeled/total_imgs*100)}%)")
    print(f"==================================================")

    if total_labeled == 0:
        print("\n[!] No labeled images found!")
        print("Please run `python annotate.py` first to draw bounding boxes around origin_3.")
        return False

    if unlabeled_images:
        print(f"[!] Warning: {len(unlabeled_images)} images are still unlabeled.")
        print("    You can proceed now, or annotate the remaining images for higher accuracy.")

    # Shuffle and split
    random.shuffle(labeled_pairs)
    train_count = max(1, int(total_labeled * split_ratio))
    train_pairs = labeled_pairs[:train_count]
    val_pairs = labeled_pairs[train_count:]

    if not val_pairs:
        # If very few images, put at least 1 in val
        val_pairs = [train_pairs[-1]]

    print(f"-> Split: {len(train_pairs)} Train images, {len(val_pairs)} Val images.")

    # Target directories
    dirs = {
        "train_img": os.path.join(YOLO_DIR, "images", "train"),
        "val_img": os.path.join(YOLO_DIR, "images", "val"),
        "train_lbl": os.path.join(YOLO_DIR, "labels", "train"),
        "val_lbl": os.path.join(YOLO_DIR, "labels", "val"),
    }

    # Reset YOLO dir
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Copy files
    for img, lbl in train_pairs:
        shutil.copy2(img, dirs["train_img"])
        shutil.copy2(lbl, dirs["train_lbl"])

    for img, lbl in val_pairs:
        shutil.copy2(img, dirs["val_img"])
        shutil.copy2(lbl, dirs["val_lbl"])

    # Create data.yaml
    data_yaml_content = {
        "path": YOLO_DIR.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: CLASS_NAME
        }
    }

    yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"\n[SUCCESS] YOLO dataset prepared at: {YOLO_DIR}")
    print(f"[SUCCESS] Configuration file saved at: {yaml_path}")
    print(f"\nYou are now ready to train! Run:\n  python train.py\n")
    return True


if __name__ == "__main__":
    prepare_yolo_dataset()
