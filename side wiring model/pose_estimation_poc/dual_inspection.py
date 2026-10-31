"""
Dual Inspection Engine:
Camera 1 (Top Camera) -> YOLOv8-Pose Wire Harness Inspection
Camera 2 (Side Camera) -> YOLOv8 Origin 3 Object Detection
Evaluates both; Overall system passes ONLY IF BOTH pass.
"""

import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)

# Path to inspect_pose in this directory
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from inspect_pose import inspect_image

# Global model caches
_POSE_MODEL = None
_ORIGIN3_MODEL = None


def get_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        p_paths = [
            os.path.join(SCRIPT_DIR, "runs", "wire_pose_model", "weights", "best.pt"),
            os.path.join(SCRIPT_DIR, "yolov8n-pose.pt"),
        ]
        chosen = None
        for p in p_paths:
            if os.path.exists(p):
                chosen = p
                break
        if chosen is None:
            chosen = "yolov8n-pose.pt"
        print(f"[DualInspection] Loading Pose model from: {chosen}")
        _POSE_MODEL = YOLO(chosen)
    return _POSE_MODEL


def get_origin3_model():
    global _ORIGIN3_MODEL
    if _ORIGIN3_MODEL is None:
        o_paths = [
            os.path.join(WORKSPACE_DIR, "runs", "origin3_model", "weights", "best.pt"),
            os.path.join(SCRIPT_DIR, "runs", "origin3_model", "weights", "best.pt"),
            os.path.join(WORKSPACE_DIR, "yolov8n.pt"),
        ]
        chosen = None
        for p in o_paths:
            if os.path.exists(p):
                chosen = p
                break
        if chosen is None:
            raise FileNotFoundError("Origin 3 model weights not found!")
        print(f"[DualInspection] Loading Origin 3 model from: {chosen}")
        _ORIGIN3_MODEL = YOLO(chosen)
    return _ORIGIN3_MODEL


def inspect_origin3(model, image_input, conf=0.4):
    """
    Runs Origin 3 object detection on side view image.
    Pass condition: Origin 3 detected with confidence >= conf.
    """
    t0 = time.time()
    if isinstance(image_input, np.ndarray):
        img = image_input
        img_name = "Side Camera Frame"
    elif isinstance(image_input, str):
        img = cv2.imread(image_input)
        img_name = os.path.basename(image_input)
    else:
        img = None
        img_name = "Unknown"

    if img is None:
        return {
            "is_pass": False,
            "confidence": 0.0,
            "bbox": None,
            "center": None,
            "msg": "Failed to load side view image for Origin 3.",
            "annotated_img": None,
            "orig_img": None,
            "latency_ms": 0.0,
        }

    results = model(img, conf=conf, verbose=False)
    r = results[0]
    boxes = r.boxes

    annotated_img = img.copy()
    is_pass = False
    best_conf = 0.0
    best_bbox = None
    best_center = None

    if boxes is not None and len(boxes) > 0:
        # Find detection with highest confidence
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = xyxy.tolist()
        best_bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        cx = round((x1 + x2) / 2.0, 1)
        cy = round((y1 + y2) / 2.0, 1)
        best_center = [cx, cy]
        is_pass = True

        # Draw on annotated image
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        icx, icy = int(cx), int(cy)

        # Bounding box
        cv2.rectangle(annotated_img, (ix1, iy1), (ix2, iy2), (0, 220, 0), 3)

        # Center marker crosshair
        cv2.drawMarker(
            annotated_img, (icx, icy), (0, 255, 255),
            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2
        )

        # Badge label
        label_text = f"origin_3 {best_conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_img, (ix1, max(0, iy1 - th - 10)), (ix1 + tw + 10, max(th + 10, iy1)), (0, 220, 0), -1)
        cv2.putText(annotated_img, label_text, (ix1 + 5, max(th + 2, iy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Center coord text
        coord_text = f"Center: ({icx}, {icy})"
        cv2.putText(annotated_img, coord_text, (ix1, min(annotated_img.shape[0] - 8, iy2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        msg = f"Origin 3 detected at ({icx}, {icy}) with {best_conf*100:.1f}% confidence."
    else:
        msg = "Origin 3 NOT detected (component missing, misaligned, or below confidence)."

    # Header badge
    badge_color = (0, 180, 0) if is_pass else (0, 0, 220)
    badge_text = f"ORIGIN 3: {'PASS' if is_pass else 'FAIL'}"
    cv2.rectangle(annotated_img, (20, 20), (320, 70), badge_color, -1)
    cv2.putText(annotated_img, badge_text, (32, 54), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    latency_ms = (time.time() - t0) * 1000.0

    return {
        "is_pass": is_pass,
        "confidence": round(best_conf, 3),
        "bbox": best_bbox,
        "center": best_center,
        "msg": msg,
        "annotated_img": annotated_img,
        "orig_img": img,
        "latency_ms": round(latency_ms, 1),
    }


def inspect_dual_system(top_image_input, origin3_image_input, conf_pose=0.35, conf_orig3=0.4):
    """
    Main Dual-Camera Inspection Pipeline.
    Evaluates both Top View (Wire Pose) and Side View (Origin 3).
    OVERALL PASS is True ONLY IF BOTH pass.
    """
    t_start = time.time()

    pose_model = get_pose_model()
    orig3_model = get_origin3_model()

    # Inspect Top View (Wire Pose)
    pose_res = inspect_image(pose_model, top_image_input, conf=conf_pose, return_details=True)

    # Inspect Side View (Origin 3)
    orig3_res = inspect_origin3(orig3_model, origin3_image_input, conf=conf_orig3)

    # Combined Verdict: BOTH must pass
    pose_pass = bool(pose_res.get("is_pass", False))
    orig3_pass = bool(orig3_res.get("is_pass", False))
    overall_pass = bool(pose_pass and orig3_pass)

    # Compile combined reasons for failure
    reasons_failed = []
    if not pose_pass:
        for r in pose_res.get("reasons_failed", []):
            reasons_failed.append(f"[Top View / Pose] {r}")
    if not orig3_pass:
        reasons_failed.append(f"[Side View / Origin 3] {orig3_res.get('msg', 'Origin 3 detection failed')}")

    # Compile combined passes
    passes = []
    for p in pose_res.get("passes", []):
        passes.append(f"[Top View / Pose] {p}")
    if orig3_pass:
        passes.append(f"[Side View / Origin 3] {orig3_res.get('msg', 'Origin 3 detected successfully')}")

    total_latency_ms = round((time.time() - t_start) * 1000.0, 1)

    return {
        "overall_pass": overall_pass,
        "pose_pass": pose_pass,
        "orig3_pass": orig3_pass,
        "total_latency_ms": total_latency_ms,
        "reasons_failed": reasons_failed,
        "passes": passes,
        "pose_result": pose_res,
        "origin3_result": orig3_res,
    }
