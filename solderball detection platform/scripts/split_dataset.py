"""
Dataset Split and Format Converter Script.
Supports:
1. YOLO format (.txt)
2. Labelme format (.json) -> auto-converts to YOLO
3. Pascal VOC format (.xml) -> auto-converts to YOLO

Usage:
    python scripts/split_dataset.py --val-ratio 0.2 --seed 42
"""

import os
import glob
import shutil
import random
import json
import xml.etree.ElementTree as ET
import cv2
import argparse

def convert_labelme_to_yolo(json_path, img_w, img_h):
    """Converts a Labelme JSON file to YOLO format lines."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    yolo_lines = []
    for shape in data.get('shapes', []):
        label = shape.get('label', 'solder_ball').strip().lower()
        # All defect labels mapped to class 0
        class_id = 0
        points = shape.get('points', [])
        if not points:
            continue
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        # Clamp to image boundaries
        xmin = max(0, min(xmin, img_w))
        xmax = max(0, min(xmax, img_w))
        ymin = max(0, min(ymin, img_h))
        ymax = max(0, min(ymax, img_h))
        
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
            
        x_center = (xmin + xmax) / (2.0 * img_w)
        y_center = (ymin + ymax) / (2.0 * img_h)
        norm_w = w / img_w
        norm_h = h / img_h
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    return yolo_lines

def convert_voc_to_yolo(xml_path, img_w, img_h):
    """Converts a Pascal VOC XML file to YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []
    
    for member in root.findall('object'):
        class_id = 0
        bndbox = member.find('bndbox')
        if bndbox is None:
            continue
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        xmin = max(0, min(xmin, img_w))
        xmax = max(0, min(xmax, img_w))
        ymin = max(0, min(ymin, img_h))
        ymax = max(0, min(ymax, img_h))
        
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
            
        x_center = (xmin + xmax) / (2.0 * img_w)
        y_center = (ymin + ymax) / (2.0 * img_h)
        norm_w = w / img_w
        norm_h = h / img_h
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    return yolo_lines

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and val sets.")
    parser.add_argument('--dataset-dir', type=str, default='dataset', help='Folder with raw images and annotations')
    parser.add_argument('--ann-dir', type=str, default=None, help='Separate annotations folder if not in dataset-dir')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio (e.g. 0.2 for 20%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible split')
    args = parser.parse_args()

    random.seed(args.seed)
    
    images = glob.glob(os.path.join(args.dataset_dir, '*.jpg')) + \
             glob.glob(os.path.join(args.dataset_dir, '*.png')) + \
             glob.glob(os.path.join(args.dataset_dir, '*.jpeg'))
             
    if not images:
        print(f"Error: No images found in '{args.dataset_dir}'.")
        return

    # Check for annotations
    search_dirs = [args.dataset_dir]
    if args.ann_dir:
        search_dirs.insert(0, args.ann_dir)
    search_dirs.append('annotations')

    matched_pairs = []
    for img_path in images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Look for corresponding .txt, .json, or .xml
        ann_found = None
        ann_type = None
        for sdir in search_dirs:
            txt_p = os.path.join(sdir, f"{base_name}.txt")
            json_p = os.path.join(sdir, f"{base_name}.json")
            xml_p = os.path.join(sdir, f"{base_name}.xml")
            if os.path.exists(txt_p):
                ann_found, ann_type = txt_p, 'yolo'
                break
            elif os.path.exists(json_p):
                ann_found, ann_type = json_p, 'labelme'
                break
            elif os.path.exists(xml_p):
                ann_found, ann_type = xml_p, 'voc'
                break
        
        if ann_found:
            matched_pairs.append((img_path, ann_found, ann_type))

    print(f"Found {len(images)} images in total.")
    print(f"Found annotations for {len(matched_pairs)} images.")

    if not matched_pairs:
        print("\nNo annotations found yet!")
        print(f"Images are ready in '{args.dataset_dir}'.")
        print("Once you annotate images (e.g., in YOLO .txt, Labelme .json, or Pascal VOC .xml format),")
        print("save the annotation files in 'dataset/' or 'annotations/' and re-run this script.")
        return

    random.shuffle(matched_pairs)
    val_count = max(1, int(len(matched_pairs) * args.val_ratio)) if len(matched_pairs) > 1 else 0
    val_pairs = matched_pairs[:val_count]
    train_pairs = matched_pairs[val_count:]

    # Create destination dirs
    os.makedirs('data/images/train', exist_ok=True)
    os.makedirs('data/images/val', exist_ok=True)
    os.makedirs('data/labels/train', exist_ok=True)
    os.makedirs('data/labels/val', exist_ok=True)

    # Clean existing data files
    for folder in ['data/images/train', 'data/images/val', 'data/labels/train', 'data/labels/val']:
        for f in glob.glob(os.path.join(folder, '*')):
            if os.path.isfile(f):
                os.remove(f)

    def process_split(pairs, split_name):
        total_balls = 0
        for img_p, ann_p, atype in pairs:
            base = os.path.basename(img_p)
            base_no_ext = os.path.splitext(base)[0]
            dst_img = os.path.join(f'data/images/{split_name}', base)
            dst_lbl = os.path.join(f'data/labels/{split_name}', f"{base_no_ext}.txt")
            
            shutil.copy2(img_p, dst_img)
            
            if atype == 'yolo':
                shutil.copy2(ann_p, dst_lbl)
                with open(ann_p, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                    total_balls += len(lines)
            else:
                # Need image dimensions for normalized conversion
                img = cv2.imread(img_p)
                ih, iw = img.shape[:2]
                if atype == 'labelme':
                    lines = convert_labelme_to_yolo(ann_p, iw, ih)
                else: # voc
                    lines = convert_voc_to_yolo(ann_p, iw, ih)
                with open(dst_lbl, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines) + "\n")
                total_balls += len(lines)
        return total_balls

    train_balls = process_split(train_pairs, 'train')
    val_balls = process_split(val_pairs, 'val')

    print("\nDataset split complete:")
    print(f"  Training set   : {len(train_pairs)} images, {train_balls} total solder balls annotated")
    print(f"  Validation set : {len(val_pairs)} images, {val_balls} total solder balls annotated")
    print(f"  Files saved to : 'data/images/' and 'data/labels/'")

if __name__ == '__main__':
    main()
