"""
Inference & Inspection Rules Engine for Wire Harness Pose Estimation
Loads trained YOLOv8-pose weights, extracts keypoints, and runs fail-proof
geometric & topological checks that DO NOT fail if the harness shifts.
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

def inspect_image(model, image_input, save_output_dir=None, conf=0.35, return_details=False):
    import time
    t0 = time.time()
    if isinstance(image_input, np.ndarray):
        img = image_input
        img_name = "Camera Frame"
    elif isinstance(image_input, str):
        img = cv2.imread(image_input)
        img_name = os.path.basename(image_input)
    else:
        img = None
        img_name = "Unknown"

    if img is None:
        print(f"[ERROR] Could not load image {image_input}")
        if return_details:
            return {
                "is_pass": False,
                "reasons_failed": ["Image load failed"],
                "passes": [],
                "annotated_img": None,
                "orig_img": None,
                "regions": {},
                "latency_ms": 0.0
            }
        return False, ["Image load failed"]

    orig_h, orig_w = img.shape[:2]

    # Run YOLO Pose model with specified confidence
    results = model(img, conf=conf, verbose=False)
    r = results[0]

    regions = {
        'UM': {'name': 'Upper Middle (Coils)', 'red': [], 'black': [], 'pass': True, 'msg': '', 'checks': {}},
        'BL': {'name': 'Below Left (Terminals)', 'red': [], 'black': [], 'pass': True, 'msg': '', 'checks': {}},
        'BR': {'name': 'Below Right (Exit / Grommet)', 'red': [], 'black': [], 'pass': True, 'msg': '', 'checks': {}}
    }

    if r.boxes is not None and r.keypoints is not None:
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy()  # shape (N, 3, 2)
        boxes = r.boxes.xyxy.cpu().numpy()

        for i, cls_id in enumerate(classes):
            wire_pts = kpts[i]
            box = boxes[i]
            cls_name = model.names.get(cls_id, f"wire_{cls_id}").lower() if hasattr(model, 'names') and model.names else ('red' if cls_id == 0 else 'black')
            cname = 'red' if 'red' in cls_name else ('black' if 'black' in cls_name else cls_name)
            mx, my = np.mean(wire_pts[:, 0]), np.mean(wire_pts[:, 1])

            # Classify into 3 topological regions
            if my < 0.45 * orig_h:
                reg = 'UM'
            elif mx < 0.55 * orig_w:
                reg = 'BL'
            else:
                reg = 'BR'

            if cname not in regions[reg]:
                regions[reg][cname] = []

            regions[reg][cname].append({'pts': wire_pts, 'box': box, 'conf': confs[i]})

    # Sort detections by confidence descending
    for reg in ['UM', 'BL', 'BR']:
        for cname in list(regions[reg].keys()):
            if isinstance(regions[reg][cname], list):
                regions[reg][cname].sort(key=lambda x: x['conf'], reverse=True)

    print(f"\n--- Inspecting: {img_name} ---")
    reasons_failed = []
    passes = []
    is_pass = True

    # Helper functions for keypoints
    def get_orig_kpt(wire_item):
        pts = wire_item['pts']
        return pts[0]

    def get_term_kpt(wire_item):
        pts = wire_item['pts']
        return pts[2] if len(pts) > 2 else pts[-1]

    # 1. UPPER MIDDLE (UM) CHECK
    # Origin Rule: Red wire's origin should be on the left of black wire's origin (X_red < X_black)
    # Terminal Rule: Red wire's socket/terminal should be on the right of black wire's socket/terminal (X_red > X_black)
    um = regions['UM']
    if len(um['red']) == 0 or len(um['black']) == 0:
        is_pass = False
        um['pass'] = False
        msg = f"Upper Middle: Missing wire (Found {len(um['red'])} Red, {len(um['black'])} Black)."
        um['msg'] = msg
        um['checks'] = {
            'origin': {'pass': False, 'name': 'Wire Origin', 'rule': 'Red origin left of Black (X_red < X_black)', 'detail': f"Missing wire (Found {len(um['red'])} Red, {len(um['black'])} Black)"},
            'terminal': {'pass': False, 'name': 'Wire Terminal / Socket', 'rule': 'Red terminal right of Black (X_red > X_black)', 'detail': f"Missing wire (Found {len(um['red'])} Red, {len(um['black'])} Black)"}
        }
        reasons_failed.append(msg)
    else:
        r_orig = get_orig_kpt(um['red'][0])
        b_orig = get_orig_kpt(um['black'][0])
        r_term = get_term_kpt(um['red'][0])
        b_term = get_term_kpt(um['black'][0])
        um['red_orig'], um['black_orig'] = r_orig, b_orig
        um['red_term'], um['black_term'] = r_term, b_term

        um_fails = []
        # Origin check
        orig_pass = bool(r_orig[0] < b_orig[0])
        if not orig_pass:
            um_fails.append(f"Origin Inverted: Red origin (X={r_orig[0]:.0f}) right of Black origin (X={b_orig[0]:.0f})")
        else:
            passes.append(f"Upper Middle Origin: Correct (Red origin X={r_orig[0]:.0f} is left of Black origin X={b_orig[0]:.0f}).")

        # Terminal check
        term_pass = bool(r_term[0] > b_term[0])
        if not term_pass:
            um_fails.append(f"Terminal Inverted: Red terminal (X={r_term[0]:.0f}) left of Black terminal (X={b_term[0]:.0f})")
        else:
            passes.append(f"Upper Middle Terminal: Correct (Red terminal X={r_term[0]:.0f} is right of Black terminal X={b_term[0]:.0f}).")

        um['checks'] = {
            'origin': {
                'pass': orig_pass,
                'name': 'Wire Origin',
                'rule': 'Red origin left of Black (X_red < X_black)',
                'detail': f'Red X={r_orig[0]:.0f} < Black X={b_orig[0]:.0f}' if orig_pass else f'Red X={r_orig[0]:.0f} >= Black X={b_orig[0]:.0f} (Inverted)'
            },
            'terminal': {
                'pass': term_pass,
                'name': 'Wire Terminal / Socket',
                'rule': 'Red terminal right of Black (X_red > X_black)',
                'detail': f'Red X={r_term[0]:.0f} > Black X={b_term[0]:.0f}' if term_pass else f'Red X={r_term[0]:.0f} <= Black X={b_term[0]:.0f} (Inverted)'
            }
        }

        if um_fails:
            is_pass = False
            um['pass'] = False
            um['msg'] = "Upper Middle: " + "; ".join(um_fails)
            reasons_failed.extend([f"Upper Middle {f}" for f in um_fails])
        else:
            um['pass'] = True
            um['msg'] = "Upper Middle: Correct (Origin: Red left of Black, Terminal: Red right of Black)."

    # 2. BOTTOM LEFT PAIR (BL) CHECK
    # Origin Rule: Red wire's origin should be below black wire's origin (Y_red > Y_black)
    # Terminal Rule: Red wire's socket/terminal should be on the left to black wire's socket/terminal (X_red < X_black)
    bl = regions['BL']
    if len(bl['red']) == 0 or len(bl['black']) == 0:
        is_pass = False
        bl['pass'] = False
        msg = f"Bottom Left: Missing wire (Found {len(bl['red'])} Red, {len(bl['black'])} Black)."
        bl['msg'] = msg
        bl['checks'] = {
            'origin': {'pass': False, 'name': 'Wire Origin', 'rule': 'Red origin below Black (Y_red > Y_black)', 'detail': f"Missing wire (Found {len(bl['red'])} Red, {len(bl['black'])} Black)"},
            'terminal': {'pass': False, 'name': 'Wire Terminal / Socket', 'rule': 'Red terminal left of Black (X_red < X_black)', 'detail': f"Missing wire (Found {len(bl['red'])} Red, {len(bl['black'])} Black)"}
        }
        reasons_failed.append(msg)
    else:
        r_orig = get_orig_kpt(bl['red'][0])
        b_orig = get_orig_kpt(bl['black'][0])
        r_term = get_term_kpt(bl['red'][0])
        b_term = get_term_kpt(bl['black'][0])
        bl['red_orig'], bl['black_orig'] = r_orig, b_orig
        bl['red_term'], bl['black_term'] = r_term, b_term

        bl_fails = []
        # Origin check
        orig_pass = bool(r_orig[1] > b_orig[1])
        if not orig_pass:
            bl_fails.append(f"Origin Inversion: Red origin (Y={r_orig[1]:.0f}) above/level with Black origin (Y={b_orig[1]:.0f})")
        else:
            passes.append(f"Bottom Left Origin: Correct (Red origin Y={r_orig[1]:.0f} is below Black origin Y={b_orig[1]:.0f}).")

        # Terminal check
        term_pass = bool(r_term[0] < b_term[0])
        if not term_pass:
            bl_fails.append(f"Terminal Inversion: Red terminal (X={r_term[0]:.0f}) right of Black terminal (X={b_term[0]:.0f})")
        else:
            passes.append(f"Bottom Left Terminal: Correct (Red terminal X={r_term[0]:.0f} is left of Black terminal X={b_term[0]:.0f}).")

        bl['checks'] = {
            'origin': {
                'pass': orig_pass,
                'name': 'Wire Origin',
                'rule': 'Red origin below Black origin (Y_red > Y_black)',
                'detail': f'Red Y={r_orig[1]:.0f} > Black Y={b_orig[1]:.0f} (Below)' if orig_pass else f'Red Y={r_orig[1]:.0f} <= Black Y={b_orig[1]:.0f} (Above/Level - Inverted)'
            },
            'terminal': {
                'pass': term_pass,
                'name': 'Wire Terminal / Socket',
                'rule': 'Red terminal left of Black terminal (X_red < X_black)',
                'detail': f'Red X={r_term[0]:.0f} < Black X={b_term[0]:.0f}' if term_pass else f'Red X={r_term[0]:.0f} >= Black X={b_term[0]:.0f} (Inverted)'
            }
        }

        if bl_fails:
            is_pass = False
            bl['pass'] = False
            bl['msg'] = "Bottom Left: " + "; ".join(bl_fails)
            reasons_failed.extend([f"Bottom Left {f}" for f in bl_fails])
        else:
            bl['pass'] = True
            bl['msg'] = "Bottom Left: Correct (Origin: Red below Black, Terminal: Red left of Black)."

    # 3. BOTTOM RIGHT PAIR (BR) CHECK
    # Origin Rule: Both wire origins should just exist
    # Terminal Rule: Red wire's socket/terminal should be on the left to black wire's socket/terminal (X_red < X_black)
    br = regions['BR']
    if len(br['red']) == 0 or len(br['black']) == 0:
        is_pass = False
        br['pass'] = False
        msg = f"Bottom Right: Missing wire (Found {len(br['red'])} Red, {len(br['black'])} Black)."
        br['msg'] = msg
        br['checks'] = {
            'origin': {'pass': False, 'name': 'Wire Origin', 'rule': 'Both wire origins must exist', 'detail': f"Missing wire (Found {len(br['red'])} Red, {len(br['black'])} Black)"},
            'terminal': {'pass': False, 'name': 'Wire Terminal / Socket', 'rule': 'Red terminal left of Black (X_red < X_black)', 'detail': f"Missing wire (Found {len(br['red'])} Red, {len(br['black'])} Black)"}
        }
        reasons_failed.append(msg)
    else:
        r_orig = get_orig_kpt(br['red'][0])
        b_orig = get_orig_kpt(br['black'][0])
        r_term = get_term_kpt(br['red'][0])
        b_term = get_term_kpt(br['black'][0])
        br['red_orig'], br['black_orig'] = r_orig, b_orig
        br['red_term'], br['black_term'] = r_term, b_term

        br_fails = []
        # Origin check: both exist
        orig_pass = True
        passes.append(f"Bottom Right Origin: Both wire origins exist (Red at ({r_orig[0]:.0f}, {r_orig[1]:.0f}), Black at ({b_orig[0]:.0f}, {b_orig[1]:.0f})).")

        # Terminal check
        term_pass = bool(r_term[0] < b_term[0])
        if not term_pass:
            br_fails.append(f"Terminal Inversion: Red terminal (X={r_term[0]:.0f}) right of Black terminal (X={b_term[0]:.0f})")
        else:
            passes.append(f"Bottom Right Terminal: Correct (Red terminal X={r_term[0]:.0f} is left of Black terminal X={b_term[0]:.0f}).")

        br['checks'] = {
            'origin': {
                'pass': orig_pass,
                'name': 'Wire Origin',
                'rule': 'Both wire origins must exist',
                'detail': f'Red at ({r_orig[0]:.0f}, {r_orig[1]:.0f}), Black at ({b_orig[0]:.0f}, {b_orig[1]:.0f})'
            },
            'terminal': {
                'pass': term_pass,
                'name': 'Wire Terminal / Socket',
                'rule': 'Red terminal left of Black (X_red < X_black)',
                'detail': f'Red X={r_term[0]:.0f} < Black X={b_term[0]:.0f}' if term_pass else f'Red X={r_term[0]:.0f} >= Black X={b_term[0]:.0f} (Inverted)'
            }
        }

        if br_fails:
            is_pass = False
            br['pass'] = False
            br['msg'] = "Bottom Right: " + "; ".join(br_fails)
            reasons_failed.extend([f"Bottom Right {f}" for f in br_fails])
        else:
            br['pass'] = True
            br['msg'] = "Bottom Right: Correct (Both origins exist, Terminal: Red left of Black)."

    for p in passes:
        print(f"  [OK] {p}")
    for f in reasons_failed:
        print(f"  [FAIL] {f}")

    # Render Visual Skeleton
    annotated_img = r.plot()

    # Highlight evaluated origins and terminals with distinctive badges
    for reg_key in ['UM', 'BL', 'BR']:
        reg = regions[reg_key]
        # Origins (Square badge)
        if 'red_orig' in reg and 'black_orig' in reg:
            ro = (int(reg['red_orig'][0]), int(reg['red_orig'][1]))
            bo = (int(reg['black_orig'][0]), int(reg['black_orig'][1]))
            # Red origin
            cv2.circle(annotated_img, ro, 6, (0, 0, 255), -1)
            cv2.circle(annotated_img, ro, 9, (255, 255, 255), 1)
            cv2.putText(annotated_img, "R-Orig", (ro[0] + 10, ro[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            # Black origin
            cv2.circle(annotated_img, bo, 6, (255, 200, 0), -1)
            cv2.circle(annotated_img, bo, 9, (255, 255, 255), 1)
            cv2.putText(annotated_img, "B-Orig", (bo[0] + 10, bo[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # Terminals (Circle badge)
        if 'red_term' in reg and 'black_term' in reg:
            rt = (int(reg['red_term'][0]), int(reg['red_term'][1]))
            bt = (int(reg['black_term'][0]), int(reg['black_term'][1]))
            # Red terminal (Red circle + white outline)
            cv2.circle(annotated_img, rt, 8, (0, 0, 255), -1)
            cv2.circle(annotated_img, rt, 11, (255, 255, 255), 2)
            cv2.putText(annotated_img, "R-Term", (rt[0] + 12, rt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Black terminal (Gold/Cyan circle + white outline)
            cv2.circle(annotated_img, bt, 8, (255, 200, 0), -1)
            cv2.circle(annotated_img, bt, 11, (255, 255, 255), 2)
            cv2.putText(annotated_img, "B-Term", (bt[0] + 12, bt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    badge_color = (0, 180, 0) if is_pass else (0, 0, 220)
    badge_text = "OVERALL: PASS" if is_pass else "OVERALL: FAIL"
    
    cv2.rectangle(annotated_img, (20, 20), (360, 80), badge_color, -1)
    cv2.putText(annotated_img, badge_text, (35, 62), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    if not is_pass:
        for idx, reason in enumerate(reasons_failed):
            y_pos = 115 + idx * 30
            # Draw semi-dark background strip for readability
            cv2.rectangle(annotated_img, (20, y_pos - 20), (min(orig_w - 20, 1200), y_pos + 8), (20, 20, 20), -1)
            cv2.putText(annotated_img, f"! {reason}", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    latency_ms = (time.time() - t0) * 1000.0

    if save_output_dir:
        os.makedirs(save_output_dir, exist_ok=True)
        out_name = os.path.basename(image_input) if isinstance(image_input, str) else "frame.jpg"
        out_path = os.path.join(save_output_dir, f"inspected_{out_name}")
        cv2.imwrite(out_path, annotated_img)
        print(f"Saved inspection result to: {out_path}")

    if return_details:
        return {
            "is_pass": is_pass,
            "passes": passes,
            "reasons_failed": reasons_failed,
            "regions": regions,
            "annotated_img": annotated_img,
            "orig_img": img,
            "latency_ms": latency_ms,
            "total_wires": len(r.boxes) if r.boxes is not None else 0
        }

    return is_pass, reasons_failed

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "runs", "wire_pose_model", "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"\n[INFO] Model weights not found at: {model_path}")
        print("Falling back to pretrained 'yolov8n-pose.pt' for POC pipeline demonstration.")
        model_path = "yolov8n-pose.pt"

    model = YOLO(model_path)
    
    # Test on images
    test_dir = os.path.join(script_dir, "test_samples")
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith((".jpg", ".png"))]

    if not test_images:
        # Fallback to check samples from validation set
        val_dir = os.path.join(script_dir, "dataset", "images", "val")
        if os.path.exists(val_dir):
            test_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith((".jpg", ".png"))][:5]

    out_dir = os.path.join(script_dir, "runs", "inspections")
    for img_path in test_images:
        inspect_image(model, img_path, save_output_dir=out_dir)

if __name__ == "__main__":
    main()
