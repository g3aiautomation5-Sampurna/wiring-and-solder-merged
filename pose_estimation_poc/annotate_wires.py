"""
Interactive Keypoint Annotation Tool for Wire Harness POC
Allows labeling keypoints for:
- Red Wire (Class 0)
- Black Wire (Class 1)

Keypoints per wire:
1. Origin (Start connection / Coil)
2. Midpoint (Grommet / Bend)
3. Terminal (Socket connection)
"""

import cv2
import os
import glob
import numpy as np

import argparse

KEYPOINT_NAMES = ["1. Origin (Start/Coil)", "2. Midpoint (Grommet/Bend)", "3. Terminal (Socket)"]

def choose_input_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Check paths
    base_dirs = [
        os.path.join(parent_dir, "Wiring Dataset"),
        r"C:\Users\SM 461\Desktop\Try 2\Wiring Dataset",
        os.path.join(script_dir, "Wiring Dataset"),
    ]
    wiring_base = None
    for b in base_dirs:
        if os.path.exists(b):
            wiring_base = b
            break

    if wiring_base is None:
        return os.path.join(parent_dir, "Wiring Dataset", "Correct Wiring")

    folders = {
        "1": ("Wrong Wiring - Case 3 (Bottom Right Inverted)", os.path.join(wiring_base, "Wrong Wiring", "case 3")),
        "2": ("Wrong Wiring - Case 1 (Bottom Left Inverted)", os.path.join(wiring_base, "Wrong Wiring", "case 1")),
        "3": ("Wrong Wiring - Case 2 (Upper Middle Inverted)", os.path.join(wiring_base, "Wrong Wiring", "case 2")),
        "4": ("Correct Wiring (Standard Assemblies)", os.path.join(wiring_base, "Correct Wiring")),
    }

    print("\n=======================================================")
    print("Wire Pose Annotation - Choose Image Folder")
    print("=======================================================")
    for key, (desc, path) in folders.items():
        exists = "✔ Available" if os.path.exists(path) else "✗ Not found"
        print(f"  [{key}] {desc} ({exists})")
    print("  [5] Enter Custom Folder Path")
    print("=======================================================")

    choice = input("\nEnter choice [1-5] (default is 1 for Case 3): ").strip()
    if choice == "" or choice == "1":
        return folders["1"][1]
    elif choice in folders:
        return folders[choice][1]
    elif choice == "5":
        custom = input("Enter full path to folder: ").strip().strip('"').strip("'")
        if os.path.exists(custom):
            return custom
        print("Folder does not exist, using default Case 3.")
        return folders["1"][1]
    return folders["1"][1]

def run_annotator(input_dir=None, output_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if input_dir is None:
        input_dir = choose_input_dir()

    if output_dir is None:
        output_dir = os.path.join(script_dir, "dataset")

    images = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    if not images:
        print(f"No images found in: {input_dir}")
        return

    train_img_dir = os.path.join(output_dir, "images", "train")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Set clean resolution that guarantees 100% of the image & box is visible on 1366x768 screens
    BANNER_H = 50
    disp_w, disp_h = 960, 540

    total_window_h = disp_h + BANNER_H  # 590px total (comfortably fits 768px height)
    window_name = "Wire Keypoint Annotator (Press Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, total_window_h)

    print(f"\n=======================================================")
    print(f"Wire Pose Annotation Tool ({len(images)} images available)")
    print(f"Display resolution: {disp_w}x{disp_h} + {BANNER_H}px top banner")
    print("Controls:")
    print("  [R] -> Set active wire to RED (Class 0)")
    print("  [B] -> Set active wire to BLACK (Class 1)")
    print("  [Left Click] -> Place Keypoint (Origin -> Midpoint -> Terminal)")
    print("  [Z] -> Undo last point/wire")
    print("  [SPACE] / [S] -> Save & Next image")
    print("  [N] -> Skip image without saving")
    print("  [Q] / [ESC] -> Quit")
    print(f"=======================================================\n")

    for img_idx, img_path in enumerate(images):
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            continue

        orig_h, orig_w = img_raw.shape[:2]
        img_disp = cv2.resize(img_raw, (disp_w, disp_h))

        # Check if already labeled
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if os.path.exists(os.path.join(train_lbl_dir, f"{base_name}.txt")) or os.path.exists(os.path.join(val_lbl_dir, f"{base_name}.txt")):
            print(f"[{img_idx+1}/{len(images)}] {base_name} already labeled, skipping...")
            continue

        current_wire_class = 0  # 0: red, 1: black
        wires = []  # list of dict: {"class": int, "points": [(x, y), (x, y), (x, y)]}
        current_points = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_points, wires, current_wire_class
            if event == cv2.EVENT_LBUTTONDOWN:
                # Coordinate mapping: subtract banner height so points correspond to image pixels
                img_y = y - BANNER_H
                img_x = x
                if 0 <= img_x < disp_w and 0 <= img_y < disp_h:
                    if len(current_points) < 3:
                        current_points.append((img_x, img_y))
                        if len(current_points) == 3:
                            wires.append({"class": current_wire_class, "points": list(current_points)})
                            current_points = []
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            # 1. Prepare image canvas (with keypoints & lines drawn)
            canvas = img_disp.copy()

            # Draw previously finished wires
            for wire in wires:
                cls_id = wire["class"]
                color = (0, 0, 255) if cls_id == 0 else (50, 50, 50)
                pts = wire["points"]
                for i, pt in enumerate(pts):
                    cv2.circle(canvas, pt, 6, color, -1)
                    cv2.circle(canvas, pt, 8, (255, 255, 255), 1)
                    cv2.putText(canvas, f"{i+1}", (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                for i in range(len(pts)-1):
                    cv2.line(canvas, pts[i], pts[i+1], color, 2)

            # Draw current in-progress points
            active_color = (0, 0, 255) if current_wire_class == 0 else (200, 200, 200)
            for i, pt in enumerate(current_points):
                cv2.circle(canvas, pt, 6, active_color, -1)
                cv2.putText(canvas, f"{i+1}", (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            for i in range(len(current_points)-1):
                cv2.line(canvas, current_points[i], current_points[i+1], active_color, 2)

            # 2. Prepare top HUD banner (dedicated separate area so image is 100% visible)
            hud_banner = np.zeros((BANNER_H, disp_w, 3), dtype=np.uint8)
            cv2.rectangle(hud_banner, (0, 0), (disp_w, BANNER_H), (25, 25, 25), -1)

            curr_name = "RED (Class 0)" if current_wire_class == 0 else "BLACK (Class 1)"
            next_pt = KEYPOINT_NAMES[len(current_points)] if len(current_points) < 3 else "Finished wire! (Select next or Save)"
            cv2.putText(hud_banner, f"Image: {os.path.basename(img_path)} [{img_idx+1}/{len(images)}] | Active: {curr_name} | Wires added: {len(wires)}", 
                        (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            cv2.putText(hud_banner, f"Next: {next_pt} | [R]ed [B]lack [Z]Undo [SPACE/S]Save [N]ext [Q]uit", 
                        (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)

            # Stack banner above the image
            full_view = np.vstack([hud_banner, canvas])

            cv2.imshow(window_name, full_view)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q') or key == 27:  # Quit
                cv2.destroyAllWindows()
                print("Annotation stopped by user.")
                return
            elif key == ord('r'):
                current_wire_class = 0
            elif key == ord('b'):
                current_wire_class = 1
            elif key == ord('z'):
                if current_points:
                    current_points.pop()
                elif wires:
                    wires.pop()
            elif key == ord('n'):  # Skip
                break
            elif key == ord('s') or key == 32:  # Save
                if len(wires) == 0:
                    print("No wires labeled. Press [N] to skip without saving.")
                    continue

                # 80/20 train/val split
                is_val = (img_idx % 5 == 0)
                target_img_dir = val_img_dir if is_val else train_img_dir
                target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

                # Save resized image (standard 1280x720)
                out_img_path = os.path.join(target_img_dir, f"{base_name}.jpg")
                cv2.imwrite(out_img_path, img_disp)

                # Generate YOLO Pose TXT lines
                lbl_path = os.path.join(target_lbl_dir, f"{base_name}.txt")
                with open(lbl_path, "w") as f:
                    for wire in wires:
                        cls_id = wire["class"]
                        pts = wire["points"]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        # Bounding box around keypoints with padding
                        pad = 20
                        bx1 = max(0, min(xs) - pad)
                        by1 = max(0, min(ys) - pad)
                        bx2 = min(disp_w, max(xs) + pad)
                        by2 = min(disp_h, max(ys) + pad)

                        bw = bx2 - bx1
                        bh = by2 - by1
                        bcx = bx1 + bw / 2.0
                        bcy = by1 + bh / 2.0

                        # Normalized bbox
                        norm_bcx = bcx / disp_w
                        norm_bcy = bcy / disp_h
                        norm_bw = bw / disp_w
                        norm_bh = bh / disp_h

                        # Normalized keypoints: x y v (v=2 visible)
                        kpts_str = []
                        for pt in pts:
                            kpts_str.append(f"{pt[0]/disp_w:.6f} {pt[1]/disp_h:.6f} 2")

                        line = f"{cls_id} {norm_bcx:.6f} {norm_bcy:.6f} {norm_bw:.6f} {norm_bh:.6f} " + " ".join(kpts_str)
                        f.write(line + "\n")

                split_type = "VAL" if is_val else "TRAIN"
                print(f"Saved {base_name}.jpg & .txt ({len(wires)} wires) -> {split_type}")
                break

    cv2.destroyAllWindows()
    print("Annotation completed or all images processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wire Keypoint Annotation Tool")
    parser.add_argument("--case", type=str, choices=["1", "2", "3", "4"], help="1: Case 3 (BR), 2: Case 1 (BL), 3: Case 2 (UM), 4: Correct")
    parser.add_argument("--input", type=str, default=None, help="Path to custom image folder")
    args = parser.parse_args()

    target_input = args.input
    if target_input is None and args.case:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        alt_base = os.path.join(parent_dir, "Wiring Dataset")
        if not os.path.exists(alt_base):
            alt_base = r"C:\Users\SM 461\Desktop\Try 2\Wiring Dataset"
        mapping = {
            "1": os.path.join(alt_base, "Wrong Wiring", "case 3"),
            "2": os.path.join(alt_base, "Wrong Wiring", "case 1"),
            "3": os.path.join(alt_base, "Wrong Wiring", "case 2"),
            "4": os.path.join(alt_base, "Correct Wiring")
        }
        target_input = mapping.get(args.case)

    run_annotator(input_dir=target_input)
