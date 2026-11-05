"""
All-in-One Vision Quality Inspection Platform
Unified Sub-System: Detects both Wiring (Top Pose & Side Origin 3) and Solder Balls (Top View).
"""

import os
import sys
import time
import json
import base64
import glob
import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify, send_file

# Resolve directory roots
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_DIR = os.path.join(ROOT_DIR, "pose_estimation_poc")
SOLDER_DIR = os.path.join(ROOT_DIR, "solderball detection platform")

if POSE_DIR not in sys.path:
    sys.path.insert(0, POSE_DIR)
if SOLDER_DIR not in sys.path:
    sys.path.insert(0, SOLDER_DIR)

from inspect_pose import inspect_image
from dual_inspection import get_pose_model, get_origin3_model, inspect_origin3
import web_platform as solder_app

app = Flask("combined_inspection_suite")

def cv2_to_base64(img, quality=92):
    if img is None:
        return ""
    success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

def decode_b64(b64_str):
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    try:
        raw_bytes = base64.b64decode(b64_str)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[CombinedInspection] Base64 decode error: {e}")
        return None

def run_combined_inspection(
    top_img,
    side_img,
    conf_pose=0.35,
    conf_orig3=0.40,
    conf_solder=0.25,
    iou_solder=0.45,
    expected_solder_count=None,
    solder_tolerance=0
):
    """
    Executes unified inspection pipeline:
    1. Top View -> Evaluated with YOLOv8-Pose for Wire Harness topological routing.
    2. Top View (SAME IMAGE) -> Evaluated with YOLOv8 Solder Ball Detector for solder balls.
    3. Side View -> Evaluated with YOLOv8 Origin 3 Detector for side alignment and presence.
    """
    t_start = time.time()

    pose_model = get_pose_model()
    orig3_model = get_origin3_model()
    solder_model = solder_app.get_or_load_model()

    # 1. Wire Harness Pose Inspection on Top View
    t0 = time.time()
    pose_res = inspect_image(pose_model, top_img, conf=conf_pose, return_details=True)
    pose_latency = round((time.time() - t0) * 1000.0, 1)

    # 2. Side View Origin 3 Inspection
    t1 = time.time()
    orig3_res = inspect_origin3(orig3_model, side_img, conf=conf_orig3)
    orig3_latency = round((time.time() - t1) * 1000.0, 1)

    # 3. Solder Ball Detection on Top View (Same Image)
    t2 = time.time()
    solder_detections = []
    if solder_model is not None:
        solder_results = solder_model.predict(top_img, conf=conf_solder, iou=iou_solder, verbose=False)
        r_solder = solder_results[0]
        if r_solder.boxes is not None and len(r_solder.boxes) > 0:
            for box in r_solder.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                solder_detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': round(conf, 3)
                })
    solder_latency = round((time.time() - t2) * 1000.0, 1)

    solder_count = len(solder_detections)

    # Evaluate Solder Ball Pass/Defect
    if expected_solder_count is not None and expected_solder_count > 0:
        solder_pass = bool(abs(solder_count - expected_solder_count) <= solder_tolerance)
        solder_msg = f"Detected {solder_count} solder balls (Target: {expected_solder_count} \u00b1 {solder_tolerance}) &mdash; {'PASS' if solder_pass else 'COUNT MISMATCH / DEFECT'}."
    else:
        solder_pass = bool(solder_count > 0)
        solder_msg = f"Detected {solder_count} solder balls on board." if solder_count > 0 else "No solder balls detected on PCB board."

    pose_pass = bool(pose_res.get("is_pass", False))
    orig3_pass = bool(orig3_res.get("is_pass", False))
    overall_pass = bool(pose_pass and orig3_pass and solder_pass)

    h_top, w_top = top_img.shape[:2]

    # --- Build Top Solder Annotated Image ---
    top_solder_annotated = top_img.copy()
    for idx, det in enumerate(solder_detections, 1):
        x1, y1, x2, y2 = det['box']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_top, x2), min(h_top, y2)

        cv2.rectangle(top_solder_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(top_solder_annotated, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (0, 0, 255), -1)

        tag = f"#{idx}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lbl_y1 = max(0, y1 - th - 6)
        cv2.rectangle(top_solder_annotated, (x1, lbl_y1), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(top_solder_annotated, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

    # Top Solder Header Banner
    banner_solder = top_solder_annotated.copy()
    cv2.rectangle(banner_solder, (0, 0), (w_top, 46), (15, 18, 26), -1)
    cv2.addWeighted(banner_solder, 0.8, top_solder_annotated, 0.2, 0, top_solder_annotated)
    s_badge_col = (0, 255, 120) if solder_pass else (0, 60, 255)
    cv2.putText(
        top_solder_annotated,
        f"SOLDER BALL INSPECTION: {solder_count} BALLS FOUND [{'PASS' if solder_pass else 'DEFECT'}]",
        (18, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, s_badge_col, 2
    )

    # --- Build Combined Annotated Top Image (Pose Skeleton + Solder Bounding Boxes) ---
    pose_annot = pose_res.get('annotated_img')
    combined_top_annotated = (pose_annot if pose_annot is not None else top_img).copy()

    for idx, det in enumerate(solder_detections, 1):
        x1, y1, x2, y2 = det['box']
        # Neon Cyan bounding box for solder balls when combined with wiring
        cv2.rectangle(combined_top_annotated, (x1, y1), (x2, y2), (255, 220, 0), 2)
        cv2.circle(combined_top_annotated, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (0, 255, 255), -1)
        # Indicator label
        lbl = f"SB#{idx}"
        cv2.putText(combined_top_annotated, lbl, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

    # Top Combined Banner overlay
    comb_banner = combined_top_annotated.copy()
    cv2.rectangle(comb_banner, (0, 0), (w_top, 52), (10, 14, 22), -1)
    cv2.addWeighted(comb_banner, 0.85, combined_top_annotated, 0.15, 0, combined_top_annotated)
    comb_col = (0, 255, 120) if (pose_pass and solder_pass) else (0, 80, 255)
    cv2.putText(
        combined_top_annotated,
        f"TOP DUAL: WIRE POSE [{'PASS' if pose_pass else 'FAIL'}]  |  SOLDER BALLS: {solder_count} [{'PASS' if solder_pass else 'DEFECT'}]",
        (18, 35), cv2.FONT_HERSHEY_DUPLEX, 0.72, comb_col, 2
    )

    # Format Component Data
    regions = pose_res.get("regions", {})
    components = []
    def make_component(cid, cname):
        reg = regions.get(cid, {})
        checks_raw = reg.get("checks", {})
        checks = []
        orig_data = checks_raw.get("origin", {})
        checks.append({
            "name": orig_data.get("name", "Wire Origin"),
            "pass": orig_data.get("pass", False),
            "rule": orig_data.get("rule", "Origin Position Check"),
            "detail": orig_data.get("detail", "Not evaluated")
        })
        term_data = checks_raw.get("terminal", {})
        checks.append({
            "name": term_data.get("name", "Wire Socket / Terminal"),
            "pass": term_data.get("pass", False),
            "rule": term_data.get("rule", "Terminal Position Check"),
            "detail": term_data.get("detail", "Not evaluated")
        })
        return {
            "id": cid,
            "name": reg.get("name", cname),
            "pass": bool(reg.get("pass", False)),
            "msg": reg.get("msg", ""),
            "checks": checks,
            "red_detected": len(reg.get("red", [])),
            "black_detected": len(reg.get("black", []))
        }

    components.append(make_component("UM", "Upper Middle (Coils)"))
    components.append(make_component("BL", "Below Left (Terminals)"))
    components.append(make_component("BR", "Below Right (Exit / Grommet)"))

    # Reasons failed and passes
    reasons_failed = []
    if not pose_pass:
        for r in pose_res.get("reasons_failed", []):
            reasons_failed.append(f"[Top View / Wire Pose] {r}")
    if not orig3_pass:
        reasons_failed.append(f"[Side View / Origin 3] {orig3_res.get('msg', 'Origin 3 detection failed')}")
    if not solder_pass:
        reasons_failed.append(f"[Top View / Solder Balls] {solder_msg}")

    passes = []
    for p in pose_res.get("passes", []):
        passes.append(f"[Top View / Wire Pose] {p}")
    if orig3_pass:
        passes.append(f"[Side View / Origin 3] {orig3_res.get('msg', 'Origin 3 detected and aligned')}")
    if solder_pass:
        passes.append(f"[Top View / Solder Balls] {solder_msg}")

    total_latency = round((time.time() - t_start) * 1000.0, 1)

    return {
        "success": True,
        "overall_pass": overall_pass,
        "pose_pass": pose_pass,
        "orig3_pass": orig3_pass,
        "solder_pass": solder_pass,
        "solder_count": solder_count,
        "solder_detections": solder_detections,
        "solder_msg": solder_msg,
        "total_latency_ms": total_latency,
        "latency_breakdown": {
            "pose_ms": pose_latency,
            "orig3_ms": orig3_latency,
            "solder_ms": solder_latency
        },
        "components": components,
        "origin3_details": {
            "is_pass": bool(orig3_res.get("is_pass", False)),
            "center": orig3_res.get("center"),
            "bbox": orig3_res.get("bbox"),
            "msg": orig3_res.get("msg", "")
        },
        "reasons_failed": reasons_failed,
        "passes": passes,
        "total_wires": pose_res.get("total_wires", 0),
        "images": {
            "top_combined": cv2_to_base64(combined_top_annotated),
            "top_pose": cv2_to_base64(pose_annot),
            "top_solder": cv2_to_base64(top_solder_annotated),
            "top_original": cv2_to_base64(top_img),
            "side_annotated": cv2_to_base64(orig3_res.get("annotated_img")),
            "side_original": cv2_to_base64(side_img)
        }
    }

COMBINED_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All-in-One Vision Quality Inspection: Wiring &amp; Solder Balls</title>
    <style>
        :root {
            --bg-master: #07090e;
            --bg-panel: #0e131d;
            --bg-card: #141b27;
            --bg-card-hover: #1c2637;
            --border: #232f45;
            --border-light: #32425e;
            --accent-purple: #8b5cf6;
            --accent-blue: #3b82f6;
            --accent-emerald: #10b981;
            --accent-rose: #ef4444;
            --accent-amber: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --pass-bg: rgba(16, 185, 129, 0.12);
            --pass-border: #10b981;
            --pass-text: #34d399;
            --fail-bg: rgba(239, 68, 68, 0.12);
            --fail-border: #ef4444;
            --fail-text: #f87171;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background: var(--bg-master);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            font-size: 13px;
        }

        /* Top Action Header */
        .station-header {
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
            padding: 8px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .station-brand {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .station-badge-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 17px;
            color: #fff;
            box-shadow: 0 0 12px rgba(139, 92, 246, 0.4);
        }

        .station-brand-text h2 {
            font-size: 0.98rem;
            font-weight: 800;
            letter-spacing: 0.4px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .station-brand-text p {
            font-size: 0.7rem;
            color: var(--text-muted);
            letter-spacing: 0.2px;
        }

        .station-pills {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .model-pill {
            display: flex;
            align-items: center;
            gap: 5px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border);
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 0.68rem;
            color: #cbd5e1;
        }

        .model-pill .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent-emerald);
            box-shadow: 0 0 6px var(--accent-emerald);
        }

        .station-actions {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .btn {
            background: #1e283b;
            color: #ffffff;
            border: 1px solid var(--border);
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 7px;
            transition: all 0.15s ease;
            outline: none;
            user-select: none;
        }

        .btn:hover {
            background: #28374f;
            border-color: var(--border-light);
        }

        .btn-upload-header {
            background: #1e293b;
            color: #93c5fd;
            border: 1px solid #3b82f6;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.25);
            cursor: pointer;
        }

        .btn-upload-header:hover {
            background: #2563eb;
            color: #ffffff;
            box-shadow: 0 0 16px rgba(59, 130, 246, 0.5);
            transform: translateY(-1px);
        }

        .btn-primary {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            border: 1px solid #4f46e5;
            box-shadow: 0 0 14px rgba(79, 70, 229, 0.4);
        }

        .btn-primary:hover {
            background: linear-gradient(135deg, #1d4ed8, #6d28d9);
            box-shadow: 0 0 18px rgba(79, 70, 229, 0.6);
            transform: translateY(-1px);
        }

        .btn-success {
            background: #065f46;
            border-color: #059669;
            color: #a7f3d0;
        }

        .btn-success:hover {
            background: #047857;
        }

        .btn-file-inline {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border);
            color: #e2e8f0;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.68rem;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            transition: all 0.15s ease;
        }

        .btn-file-inline:hover {
            background: rgba(59, 130, 246, 0.25);
            border-color: var(--accent-blue);
            color: #ffffff;
        }

        /* Main Workspace Grid */
        .workspace-grid {
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            flex: 1;
        }

        /* Master Result Banner */
        .master-result-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 14px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
            position: relative;
            overflow: hidden;
        }

        .master-result-card::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 5px;
            background: var(--accent-blue);
            transition: background 0.3s ease;
        }

        .master-result-card.verdict-pass::before {
            background: var(--accent-emerald);
        }

        .master-result-card.verdict-fail::before {
            background: var(--accent-rose);
        }

        .verdict-main {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .verdict-icon {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: 800;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
        }

        .verdict-pass .verdict-icon {
            background: rgba(16, 185, 129, 0.15);
            border-color: var(--accent-emerald);
            color: var(--accent-emerald);
            box-shadow: 0 0 16px rgba(16, 185, 129, 0.3);
        }

        .verdict-fail .verdict-icon {
            background: rgba(239, 68, 68, 0.15);
            border-color: var(--accent-rose);
            color: var(--accent-rose);
            box-shadow: 0 0 16px rgba(239, 68, 68, 0.3);
        }

        .verdict-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.4px;
        }

        .verdict-pass .verdict-title {
            color: #34d399;
        }

        .verdict-fail .verdict-title {
            color: #f87171;
        }

        .verdict-sub {
            font-size: 0.74rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .verdict-subsystems {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .subsystem-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            background: #192233;
            border: 1px solid var(--border);
            padding: 5px 12px;
            border-radius: 7px;
            font-size: 0.74rem;
            font-weight: 700;
        }

        .badge-status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
        }

        .status-dot-pass {
            background: var(--accent-emerald);
            box-shadow: 0 0 6px var(--accent-emerald);
        }

        .status-dot-fail {
            background: var(--accent-rose);
            box-shadow: 0 0 6px var(--accent-rose);
        }

        .verdict-metrics {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .metric-box {
            text-align: right;
            border-left: 1px solid var(--border);
            padding-left: 14px;
        }

        .metric-val {
            font-size: 1.05rem;
            font-weight: 800;
            color: #ffffff;
            font-family: monospace;
        }

        .metric-lbl {
            font-size: 0.64rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        /* Two Column Inspection Viewports */
        .inspection-panels-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        @media (max-width: 1040px) {
            .inspection-panels-row {
                grid-template-columns: 1fr;
            }
        }

        .view-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        }

        .view-card-header {
            background: #111723;
            border-bottom: 1px solid var(--border);
            padding: 8px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            flex-wrap: wrap;
        }

        .view-card-title {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .view-card-title h3 {
            font-size: 0.86rem;
            font-weight: 800;
            letter-spacing: 0.3px;
        }

        .view-card-title .tag {
            font-size: 0.65rem;
            padding: 2px 7px;
            border-radius: 4px;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #93c5fd;
            font-weight: 600;
        }

        .view-mode-tabs {
            display: flex;
            gap: 4px;
            background: #090d15;
            padding: 3px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }

        .view-mode-btn {
            background: transparent;
            color: var(--text-muted);
            border: none;
            padding: 4px 9px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .view-mode-btn:hover {
            color: #ffffff;
        }

        .view-mode-btn.active {
            background: #2563eb;
            color: #ffffff;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.4);
        }

        .viewport-body {
            position: relative;
            background: #090c12;
            height: 380px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border: 2px dashed transparent;
            transition: all 0.2s ease;
        }

        .viewport-body.drag-active {
            border-color: #3b82f6 !important;
            background: rgba(59, 130, 246, 0.12) !important;
        }

        .viewport-image {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transition: transform 0.2s ease;
            display: none;
        }

        .viewport-placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 10px;
            text-align: center;
            padding: 24px;
            cursor: pointer;
            user-select: none;
            transition: all 0.2s ease;
        }
        .viewport-placeholder:hover {
            transform: scale(1.02);
        }
        .viewport-placeholder .ph-icon {
            font-size: 2.4rem;
            opacity: 0.75;
        }
        .viewport-placeholder .ph-title {
            font-size: 0.9rem;
            font-weight: 700;
            color: #cbd5e1;
            letter-spacing: 0.3px;
        }
        .viewport-placeholder .ph-sub {
            font-size: 0.74rem;
            color: #64748b;
            max-width: 260px;
            line-height: 1.4;
        }

        .viewport-video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: none;
        }

        .viewport-overlay-controls {
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 6px;
            background: rgba(14, 19, 29, 0.85);
            backdrop-filter: blur(6px);
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid var(--border);
            z-index: 10;
        }

        .viewport-overlay-controls button {
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 2px 6px;
            font-size: 0.72rem;
            border-radius: 3px;
            transition: color 0.15s;
        }

        .viewport-overlay-controls button:hover {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.1);
        }

        .viewport-status-pill {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(14, 19, 29, 0.85);
            backdrop-filter: blur(6px);
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid var(--border);
            font-size: 0.7rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 6px;
            z-index: 10;
        }

        .view-card-footer {
            background: #111723;
            border-top: 1px solid var(--border);
            padding: 8px 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            flex-wrap: wrap;
        }

        .cam-controls {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }

        .select-input {
            background: #1a2333;
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 4px 8px;
            border-radius: 5px;
            font-size: 0.7rem;
            outline: none;
            max-width: 170px;
        }

        .file-upload-label {
            background: #1a2333;
            color: #cbd5e1;
            border: 1px solid var(--border);
            padding: 4px 10px;
            border-radius: 5px;
            font-size: 0.7rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: all 0.15s;
        }

        .file-upload-label:hover {
            background: #243147;
            color: #ffffff;
        }

        /* 3-Card Detailed Subsystem Breakdown */
        .subsystems-row {
            display: grid;
            grid-template-columns: 1.2fr 0.9fr 0.9fr;
            gap: 14px;
        }

        @media (max-width: 1100px) {
            .subsystems-row {
                grid-template-columns: 1fr;
            }
        }

        .subsystem-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        }

        .subsystem-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }

        .subsystem-card-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.84rem;
            font-weight: 800;
        }

        .badge-pill {
            font-size: 0.65rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }

        .badge-pill-pass {
            background: var(--pass-bg);
            color: var(--pass-text);
            border: 1px solid var(--pass-border);
        }

        .badge-pill-fail {
            background: var(--fail-bg);
            color: var(--fail-text);
            border: 1px solid var(--fail-border);
        }

        /* Diagnostic Tables & Rows */
        .diag-item {
            background: #101622;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 7px 10px;
            display: flex;
            flex-direction: column;
            gap: 3px;
        }

        .diag-item-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.74rem;
            font-weight: 700;
        }

        .diag-subtext {
            font-size: 0.68rem;
            color: var(--text-muted);
            line-height: 1.25;
        }

        .stat-grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .stat-card-mini {
            background: #101622;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 10px;
            text-align: center;
        }

        .stat-card-mini .val {
            font-size: 1.1rem;
            font-weight: 800;
            color: #ffffff;
            font-family: monospace;
        }

        .stat-card-mini .lbl {
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 2px;
        }

        /* Diagnostic Reasons Accordion */
        .diag-log-box {
            background: #0b0f17;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            max-height: 120px;
            overflow-y: auto;
            font-size: 0.7rem;
            line-height: 1.45;
        }

        .diag-pass-entry {
            color: #34d399;
            display: flex;
            align-items: flex-start;
            gap: 5px;
            margin-bottom: 3px;
        }

        .diag-fail-entry {
            color: #f87171;
            display: flex;
            align-items: flex-start;
            gap: 5px;
            margin-bottom: 3px;
            font-weight: 600;
        }

        /* Hidden snap canvas */
        canvas.hidden-canvas {
            display: none;
        }
    </style>
</head>
<body>
    <canvas id="hiddenSnapTop" class="hidden-canvas"></canvas>
    <canvas id="hiddenSnapSide" class="hidden-canvas"></canvas>

    <!-- Top Station Header -->
    <header class="station-header">
        <div class="station-brand">
            <div class="station-badge-icon">⚡</div>
            <div class="station-brand-text">
                <h2>All-in-One Quality Inspection Station</h2>
                <p>Dual-View Wire Harness (Top Pose + Side Origin 3) &amp; Top-View Solder Ball Detection</p>
            </div>
        </div>

        <div class="station-pills">
            <div class="model-pill" title="YOLOv8-Pose Top Harness Model">
                <span class="dot"></span>
                <span>Pose: Loaded</span>
            </div>
            <div class="model-pill" title="YOLOv8 Side Origin 3 Detector">
                <span class="dot"></span>
                <span>Origin 3: Loaded</span>
            </div>
            <div class="model-pill" title="YOLOv8 Solder Ball Detection Model">
                <span class="dot"></span>
                <span>Solder: Loaded</span>
            </div>
        </div>

        <div class="station-actions">
            <button class="btn" style="background:#1e293b; border:1px solid #334155; color:#cbd5e1;" onclick="loadDefaultInitialSample()" title="Load test reference images (Correct Wiring + Origin 3 Pass)">
                <span>🧪</span>
                <span>Load Sample Pair</span>
            </button>
            <label class="btn btn-upload-header" title="Upload Top and Side images from your computer (select 1 or 2 files)">
                <span>📁</span>
                <span>Upload Images</span>
                <input type="file" id="fileInputDual" accept="image/*" multiple style="display:none;" onchange="handleDualFileUpload(this)">
            </label>
            <button class="btn btn-primary" id="btnSnapInspectAll" onclick="snapAndInspectDualCameras()" title="Capture frames from both cameras and execute all 3 model evaluations">
                <span>📸</span>
                <span>Snap &amp; Inspect Both Cameras</span>
            </button>
            <button class="btn btn-success" id="btnRunCurrent" onclick="executeCombinedInspection()" title="Run inspection on current top and side images">
                <span>⚡</span>
                <span>Inspect Current Images</span>
            </button>
        </div>
    </header>

    <!-- Main Workspace -->
    <main class="workspace-grid">
        <!-- Master Result Card -->
        <div class="master-result-card" id="masterVerdictCard">
            <div class="verdict-main">
                <div class="verdict-icon" id="masterVerdictIcon">🏭</div>
                <div>
                    <div class="verdict-title" id="masterVerdictTitle">READY FOR INSPECTION</div>
                    <div class="verdict-sub" id="masterVerdictSub">Snap live cameras, click "Upload Images", or drop files onto the viewports to inspect.</div>
                </div>
            </div>

            <div class="verdict-subsystems">
                <div class="subsystem-badge" id="badgePoseSubsystem">
                    <div class="badge-status-dot status-dot-pass" id="dotPose"></div>
                    <span>Top Wire Pose:</span>
                    <strong id="txtPoseVerdict">READY</strong>
                </div>
                <div class="subsystem-badge" id="badgeOrig3Subsystem">
                    <div class="badge-status-dot status-dot-pass" id="dotOrig3"></div>
                    <span>Side Origin 3:</span>
                    <strong id="txtOrig3Verdict">READY</strong>
                </div>
                <div class="subsystem-badge" id="badgeSolderSubsystem">
                    <div class="badge-status-dot status-dot-pass" id="dotSolder"></div>
                    <span>Top Solder Balls:</span>
                    <strong id="txtSolderVerdict">READY</strong>
                </div>
            </div>

            <div class="verdict-metrics">
                <div class="metric-box">
                    <div class="metric-val" id="metricTotalLatency">-- ms</div>
                    <div class="metric-lbl">Total Latency</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" id="metricSolderCount">--</div>
                    <div class="metric-lbl">Solder Balls</div>
                </div>
            </div>
        </div>

        <!-- Two Column Viewports (Top Shared View vs Side Origin 3 View) -->
        <div class="inspection-panels-row">
            <!-- Left Viewport: Top Camera (Shared Wire Pose & Solder Balls) -->
            <div class="view-card">
                <div class="view-card-header">
                    <div class="view-card-title">
                        <h3>Camera 1: Top View</h3>
                        <span class="tag">Shared: Wire Pose + Solder Balls</span>
                        <label class="btn-file-inline" title="Upload Top View Image (Harness &amp; Solder Balls)">
                            <span>📁 Upload Top</span>
                            <input type="file" accept="image/*" style="display:none;" onchange="handleTopFileUpload(this)">
                        </label>
                    </div>

                    <div class="view-mode-tabs">
                        <button class="view-mode-btn active" id="btnModeTopCombined" onclick="switchTopViewMode('combined')">🔀 Combined</button>
                        <button class="view-mode-btn" id="btnModeTopPose" onclick="switchTopViewMode('pose')">🔌 Wire Pose</button>
                        <button class="view-mode-btn" id="btnModeTopSolder" onclick="switchTopViewMode('solder')">🔬 Solder Balls</button>
                        <button class="view-mode-btn" id="btnModeTopRaw" onclick="switchTopViewMode('raw')">📷 Raw</button>
                    </div>
                </div>

                <div class="viewport-body" id="viewportTop" title="Drop Top View Image here or click Upload Top">
                    <div class="viewport-status-pill" id="pillTopStatus">
                        <span>Top Camera</span>
                    </div>

                    <div class="viewport-placeholder" id="placeholderTop" onclick="document.getElementById('fileInputTop').click()">
                        <div class="ph-icon">📷</div>
                        <div class="ph-title">Awaiting Top View Image</div>
                        <div class="ph-sub">Drop Top image here, click "Upload Top", or start Cam 1</div>
                    </div>

                    <img id="imgTopDisplay" class="viewport-image" src="" alt="Top View Display">
                    <video id="videoTop" class="viewport-video" autoplay playsinline muted></video>

                    <div class="viewport-overlay-controls">
                        <button onclick="zoomImage('imgTopDisplay', 1.2)" title="Zoom In">+</button>
                        <button onclick="zoomImage('imgTopDisplay', 0.8)" title="Zoom Out">-</button>
                        <button onclick="resetZoom('imgTopDisplay')" title="Reset Zoom">1:1</button>
                        <button onclick="downloadActiveTopImage()" title="Download Annotated Image">📥</button>
                    </div>
                </div>

                <div class="view-card-footer">
                    <div class="cam-controls">
                        <select id="selectCamTop" class="select-input" title="Select Top Camera Device">
                            <option value="">Camera 1 (Top)...</option>
                        </select>
                        <button class="btn" style="padding:4px 8px; font-size:0.7rem;" id="btnToggleCamTop" onclick="toggleCamTop()">
                            📹 Start Cam 1
                        </button>
                        <button class="btn" style="padding:4px 8px; font-size:0.7rem;" onclick="snapTopCamera()">
                            📸 Snap Cam 1
                        </button>
                    </div>

                    <label class="file-upload-label" title="Upload custom Top View Image">
                        <span>📁 Browse Top Image</span>
                        <input type="file" id="fileInputTop" accept="image/*" style="display:none;" onchange="handleTopFileUpload(this)">
                    </label>
                </div>
            </div>

            <!-- Right Viewport: Side Camera (Origin 3 Detection) -->
            <div class="view-card">
                <div class="view-card-header">
                    <div class="view-card-title">
                        <h3>Camera 2: Side View</h3>
                        <span class="tag" style="background:rgba(16, 185, 129, 0.15); border-color:rgba(16, 185, 129, 0.3); color:#6ee7b7;">Origin 3 Detection</span>
                        <label class="btn-file-inline" title="Upload Side View Image (Origin 3)">
                            <span>📁 Upload Side</span>
                            <input type="file" accept="image/*" style="display:none;" onchange="handleSideFileUpload(this)">
                        </label>
                    </div>

                    <div class="view-mode-tabs">
                        <button class="view-mode-btn active" id="btnModeSideAnnotated" onclick="switchSideViewMode('annotated')">📐 Origin 3 Detection</button>
                        <button class="view-mode-btn" id="btnModeSideRaw" onclick="switchSideViewMode('raw')">📷 Raw Side</button>
                    </div>
                </div>

                <div class="viewport-body" id="viewportSide" title="Drop Side View Image here or click Upload Side">
                    <div class="viewport-status-pill" id="pillSideStatus">
                        <span>Side Camera</span>
                    </div>

                    <div class="viewport-placeholder" id="placeholderSide" onclick="document.getElementById('fileInputSide').click()">
                        <div class="ph-icon">📷</div>
                        <div class="ph-title">Awaiting Side View Image</div>
                        <div class="ph-sub">Drop Side image here, click "Upload Side", or start Cam 2</div>
                    </div>

                    <img id="imgSideDisplay" class="viewport-image" src="" alt="Side View Display">
                    <video id="videoSide" class="viewport-video" autoplay playsinline muted></video>

                    <div class="viewport-overlay-controls">
                        <button onclick="zoomImage('imgSideDisplay', 1.2)" title="Zoom In">+</button>
                        <button onclick="zoomImage('imgSideDisplay', 0.8)" title="Zoom Out">-</button>
                        <button onclick="resetZoom('imgSideDisplay')" title="Reset Zoom">1:1</button>
                        <button onclick="downloadActiveSideImage()" title="Download Annotated Image">📥</button>
                    </div>
                </div>

                <div class="view-card-footer">
                    <div class="cam-controls">
                        <select id="selectCamSide" class="select-input" title="Select Side Camera Device">
                            <option value="">Camera 2 (Side)...</option>
                        </select>
                        <button class="btn" style="padding:4px 8px; font-size:0.7rem;" id="btnToggleCamSide" onclick="toggleCamSide()">
                            📹 Start Cam 2
                        </button>
                        <button class="btn" style="padding:4px 8px; font-size:0.7rem;" onclick="snapSideCamera()">
                            📸 Snap Cam 2
                        </button>
                    </div>

                    <label class="file-upload-label" title="Upload custom Side View Image">
                        <span>📁 Browse Side Image</span>
                        <input type="file" id="fileInputSide" accept="image/*" style="display:none;" onchange="handleSideFileUpload(this)">
                    </label>
                </div>
            </div>
        </div>

        <!-- 3-Card Detailed Subsystem Diagnostics -->
        <div class="subsystems-row">
            <!-- Subsystem 1: Wire Harness Pose Breakdown -->
            <div class="subsystem-card">
                <div class="subsystem-card-header">
                    <div class="subsystem-card-title">
                        <span>🔌</span>
                        <span>Wire Harness Top Pose</span>
                    </div>
                    <span class="badge-pill badge-pill-pass" id="badgeCardPose">READY</span>
                </div>

                <div id="poseRegionsList" style="display:flex; flex-direction:column; gap:6px;">
                    <div class="diag-item">
                        <div class="diag-item-head">
                            <span>Upper Middle (Coils)</span>
                            <span style="color:var(--text-muted);">--</span>
                        </div>
                        <div class="diag-subtext">Waiting for inspection...</div>
                    </div>
                    <div class="diag-item">
                        <div class="diag-item-head">
                            <span>Below Left (Terminals)</span>
                            <span style="color:var(--text-muted);">--</span>
                        </div>
                        <div class="diag-subtext">Waiting for inspection...</div>
                    </div>
                    <div class="diag-item">
                        <div class="diag-item-head">
                            <span>Below Right (Exit / Grommet)</span>
                            <span style="color:var(--text-muted);">--</span>
                        </div>
                        <div class="diag-subtext">Waiting for inspection...</div>
                    </div>
                </div>
            </div>

            <!-- Subsystem 2: Side Origin 3 Breakdown -->
            <div class="subsystem-card">
                <div class="subsystem-card-header">
                    <div class="subsystem-card-title">
                        <span>📐</span>
                        <span>Side View Origin 3</span>
                    </div>
                    <span class="badge-pill badge-pill-pass" id="badgeCardOrig3">READY</span>
                </div>

                <div class="stat-grid-2">
                    <div class="stat-card-mini">
                        <div class="val" id="dispOrig3Status">READY</div>
                        <div class="lbl">Component Status</div>
                    </div>
                    <div class="stat-card-mini">
                        <div class="val" id="dispOrig3Coord">--</div>
                        <div class="lbl">Center (X, Y)</div>
                    </div>
                </div>

                <div class="diag-item">
                    <div class="diag-item-head">
                        <span>Side Alignment Check</span>
                        <span id="dispOrig3StatusText" style="color:var(--text-muted);">--</span>
                    </div>
                    <div class="diag-subtext" id="dispOrig3Msg">Origin 3 side object detection status will display here.</div>
                </div>
            </div>

            <!-- Subsystem 3: Solder Ball Breakdown -->
            <div class="subsystem-card">
                <div class="subsystem-card-header">
                    <div class="subsystem-card-title">
                        <span>🔬</span>
                        <span>Top View Solder Balls</span>
                    </div>
                    <span class="badge-pill badge-pill-pass" id="badgeCardSolder">READY</span>
                </div>

                <div class="stat-grid-2">
                    <div class="stat-card-mini">
                        <div class="val" id="dispSolderCount">--</div>
                        <div class="lbl">Detected Count</div>
                    </div>
                    <div class="stat-card-mini">
                        <div class="val" id="dispSolderQuality">READY</div>
                        <div class="lbl">Quality Verdict</div>
                    </div>
                </div>

                <div class="diag-item">
                    <div class="diag-item-head">
                        <span>Ball Count Compliance</span>
                        <span id="dispSolderCompliance" style="color:var(--text-muted);">--</span>
                    </div>
                    <div class="diag-subtext" id="dispSolderMsg">Solder ball defect evaluation will display here.</div>
                </div>
            </div>
        </div>

        <!-- Full Diagnostics & Verification Log -->
        <div class="subsystem-card">
            <div class="subsystem-card-header">
                <div class="subsystem-card-title">
                    <span>📋</span>
                    <span>Executive Verification &amp; Failure Diagnostics Log</span>
                </div>
                <span style="font-size:0.7rem; color:var(--text-muted);" id="dispLatencySummary">Latency Breakdown: --</span>
            </div>

            <div class="diag-log-box" id="diagLogBox">
                <div style="color:var(--text-muted);">No inspection run yet. Trigger an inspection to see full rule evaluations.</div>
            </div>
        </div>
    </main>

    <script>
        // State variables
        let currentTopB64 = '';
        let currentSideB64 = '';
        let inspectionResult = null;

        let activeTopMode = 'combined'; // 'combined', 'pose', 'solder', 'raw'
        let activeSideMode = 'annotated'; // 'annotated', 'raw'

        // Video streams
        let streamTop = null;
        let streamSide = null;

        // Image zoom states
        const zoomState = {
            imgTopDisplay: 1.0,
            imgSideDisplay: 1.0
        };

        // Camera enumeration & management
        async function listCameraDevices() {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(d => d.kind === 'videoinput');
                const selTop = document.getElementById('selectCamTop');
                const selSide = document.getElementById('selectCamSide');

                selTop.innerHTML = '<option value="">Select Top Camera...</option>';
                selSide.innerHTML = '<option value="">Select Side Camera...</option>';

                videoDevices.forEach((dev, idx) => {
                    const label = dev.label || `Camera ${idx + 1}`;
                    const opt1 = document.createElement('option');
                    opt1.value = dev.deviceId;
                    opt1.innerText = label;
                    selTop.appendChild(opt1);

                    const opt2 = document.createElement('option');
                    opt2.value = dev.deviceId;
                    opt2.innerText = label;
                    selSide.appendChild(opt2);
                });

                if (videoDevices.length > 0) selTop.selectedIndex = 1;
                if (videoDevices.length > 1) selSide.selectedIndex = 2;
                else if (videoDevices.length > 0) selSide.selectedIndex = 1;
            } catch (e) {
                console.log("Camera enumeration error:", e);
            }
        }

        async function toggleCamTop() {
            const video = document.getElementById('videoTop');
            const btn = document.getElementById('btnToggleCamTop');
            if (streamTop) {
                streamTop.getTracks().forEach(t => t.stop());
                streamTop = null;
                video.style.display = 'none';
                btn.innerHTML = '📹 Start Cam 1';
                btn.classList.remove('btn-success');
            } else {
                try {
                    const devId = document.getElementById('selectCamTop').value;
                    const constraints = {
                        video: devId ? { deviceId: { exact: devId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
                                     : { width: { ideal: 1920 }, height: { ideal: 1080 } }
                    };
                    streamTop = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = streamTop;
                    video.style.display = 'block';
                    const phT = document.getElementById('placeholderTop');
                    if (phT) phT.style.display = 'none';
                    btn.innerHTML = '⏹ Stop Cam 1';
                    btn.classList.add('btn-success');
                } catch (err) {
                    alert('Could not start Camera 1: ' + err.message);
                }
            }
        }

        async function toggleCamSide() {
            const video = document.getElementById('videoSide');
            const btn = document.getElementById('btnToggleCamSide');
            if (streamSide) {
                streamSide.getTracks().forEach(t => t.stop());
                streamSide = null;
                video.style.display = 'none';
                btn.innerHTML = '📹 Start Cam 2';
                btn.classList.remove('btn-success');
            } else {
                try {
                    const devId = document.getElementById('selectCamSide').value;
                    const constraints = {
                        video: devId ? { deviceId: { exact: devId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
                                     : { width: { ideal: 1920 }, height: { ideal: 1080 } }
                    };
                    streamSide = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = streamSide;
                    video.style.display = 'block';
                    const phS = document.getElementById('placeholderSide');
                    if (phS) phS.style.display = 'none';
                    btn.innerHTML = '⏹ Stop Cam 2';
                    btn.classList.add('btn-success');
                } catch (err) {
                    alert('Could not start Camera 2: ' + err.message);
                }
            }
        }

        function snapTopCamera() {
            const video = document.getElementById('videoTop');
            if (!streamTop || video.videoWidth === 0) {
                alert("Camera 1 is not running. Click 'Start Cam 1' first.");
                return;
            }
            const canvas = document.getElementById('hiddenSnapTop');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            currentTopB64 = canvas.toDataURL('image/jpeg', 0.94);
            const imgT = document.getElementById('imgTopDisplay');
            imgT.src = currentTopB64;
            imgT.style.display = 'block';
            video.style.display = 'none';
            const phT = document.getElementById('placeholderTop');
            if (phT) phT.style.display = 'none';
            document.getElementById('pillTopStatus').innerHTML = '<span>Cam 1 Frame Captured</span>';
        }

        function snapSideCamera() {
            const video = document.getElementById('videoSide');
            if (!streamSide || video.videoWidth === 0) {
                alert("Camera 2 is not running. Click 'Start Cam 2' first.");
                return;
            }
            const canvas = document.getElementById('hiddenSnapSide');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            currentSideB64 = canvas.toDataURL('image/jpeg', 0.94);
            const imgS = document.getElementById('imgSideDisplay');
            imgS.src = currentSideB64;
            imgS.style.display = 'block';
            video.style.display = 'none';
            const phS = document.getElementById('placeholderSide');
            if (phS) phS.style.display = 'none';
            document.getElementById('pillSideStatus').innerHTML = '<span>Cam 2 Frame Captured</span>';
        }

        async function snapAndInspectDualCameras() {
            const videoTop = document.getElementById('videoTop');
            const videoSide = document.getElementById('videoSide');

            if (streamTop && videoTop.videoWidth > 0) {
                snapTopCamera();
            }
            if (streamSide && videoSide.videoWidth > 0) {
                snapSideCamera();
            }

            if (!currentTopB64 || !currentSideB64) {
                alert("Please ensure both Camera 1 (Top) and Camera 2 (Side) are running, or click 'Upload Images' to choose board images.");
                return;
            }

            executeCombinedInspection();
        }

        // ================= IMAGE UPLOAD HANDLERS =================

        function readFileAndSetTop(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                currentTopB64 = e.target.result;
                const img = document.getElementById('imgTopDisplay');
                img.src = currentTopB64;
                img.style.display = 'block';
                const ph = document.getElementById('placeholderTop');
                if (ph) ph.style.display = 'none';
                document.getElementById('pillTopStatus').innerHTML = `<span>Top Image: ${file.name}</span>`;
                if (currentSideB64) {
                    executeCombinedInspection();
                }
            };
            reader.readAsDataURL(file);
        }

        function readFileAndSetSide(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                currentSideB64 = e.target.result;
                const img = document.getElementById('imgSideDisplay');
                img.src = currentSideB64;
                img.style.display = 'block';
                const ph = document.getElementById('placeholderSide');
                if (ph) ph.style.display = 'none';
                document.getElementById('pillSideStatus').innerHTML = `<span>Side Image: ${file.name}</span>`;
                if (currentTopB64) {
                    executeCombinedInspection();
                }
            };
            reader.readAsDataURL(file);
        }

        function handleTopFileUpload(input) {
            if (input.files && input.files[0]) {
                readFileAndSetTop(input.files[0]);
            }
        }

        function handleSideFileUpload(input) {
            if (input.files && input.files[0]) {
                readFileAndSetSide(input.files[0]);
            }
        }

        function handleDualFileUpload(input) {
            if (input.files && input.files.length > 0) {
                handleDualFileList(input.files);
            }
        }

        function handleDualFileList(files) {
            if (files.length === 1) {
                readFileAndSetTop(files[0]);
            } else if (files.length >= 2) {
                let file1 = files[0];
                let file2 = files[1];
                let f1Name = file1.name.toLowerCase();
                let f2Name = file2.name.toLowerCase();

                let topFile = file1;
                let sideFile = file2;

                if (f1Name.includes('side') || f1Name.includes('orig') || f1Name.includes('cam2')) {
                    sideFile = file1;
                    topFile = file2;
                } else if (f2Name.includes('top') || f2Name.includes('wire') || f2Name.includes('pose') || f2Name.includes('solder') || f2Name.includes('cam1')) {
                    topFile = file2;
                    sideFile = file1;
                }

                const readerTop = new FileReader();
                readerTop.onload = (e1) => {
                    currentTopB64 = e1.target.result;
                    const imgT = document.getElementById('imgTopDisplay');
                    imgT.src = currentTopB64;
                    imgT.style.display = 'block';
                    const phT = document.getElementById('placeholderTop');
                    if (phT) phT.style.display = 'none';
                    document.getElementById('pillTopStatus').innerHTML = `<span>Top Image: ${topFile.name}</span>`;

                    const readerSide = new FileReader();
                    readerSide.onload = (e2) => {
                        currentSideB64 = e2.target.result;
                        const imgS = document.getElementById('imgSideDisplay');
                        imgS.src = currentSideB64;
                        imgS.style.display = 'block';
                        const phS = document.getElementById('placeholderSide');
                        if (phS) phS.style.display = 'none';
                        document.getElementById('pillSideStatus').innerHTML = `<span>Side Image: ${sideFile.name}</span>`;
                        executeCombinedInspection();
                    };
                    readerSide.readAsDataURL(sideFile);
                };
                readerTop.readAsDataURL(topFile);
            }
        }

        function setupDragAndDrop() {
            const vpTop = document.getElementById('viewportTop');
            const vpSide = document.getElementById('viewportSide');

            [vpTop, vpSide].forEach((vp, idx) => {
                vp.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    vp.classList.add('drag-active');
                });
                vp.addEventListener('dragleave', (e) => {
                    e.preventDefault();
                    vp.classList.remove('drag-active');
                });
                vp.addEventListener('drop', (e) => {
                    e.preventDefault();
                    vp.classList.remove('drag-active');
                    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                        if (e.dataTransfer.files.length >= 2) {
                            handleDualFileList(e.dataTransfer.files);
                        } else {
                            const file = e.dataTransfer.files[0];
                            if (idx === 0) {
                                readFileAndSetTop(file);
                            } else {
                                readFileAndSetSide(file);
                            }
                        }
                    }
                });
            });
        }

        async function loadDefaultInitialSample() {
            try {
                const topName = 'correctwiring (1).jpg';
                const sideName = 'orig3_pass.jpg';

                const resTop = await fetch(`/api/combined/sample_image/top/${encodeURIComponent(topName)}`);
                const blobTop = await resTop.blob();
                const readerTop = new FileReader();
                readerTop.onload = (e) => {
                    currentTopB64 = e.target.result;
                    const imgT = document.getElementById('imgTopDisplay');
                    imgT.src = currentTopB64;
                    imgT.style.display = 'block';
                    const phT = document.getElementById('placeholderTop');
                    if (phT) phT.style.display = 'none';
                    document.getElementById('pillTopStatus').innerHTML = `<span>Top Board: ${topName}</span>`;

                    fetchSideInitial(sideName);
                };
                readerTop.readAsDataURL(blobTop);
            } catch (e) {
                console.log("Initial sample preview loading notice:", e);
            }
        }

        async function fetchSideInitial(sideName) {
            try {
                const resSide = await fetch(`/api/combined/sample_image/side/${encodeURIComponent(sideName)}`);
                const blobSide = await resSide.blob();
                const readerSide = new FileReader();
                readerSide.onload = (e) => {
                    currentSideB64 = e.target.result;
                    const imgS = document.getElementById('imgSideDisplay');
                    imgS.src = currentSideB64;
                    imgS.style.display = 'block';
                    const phS = document.getElementById('placeholderSide');
                    if (phS) phS.style.display = 'none';
                    document.getElementById('pillSideStatus').innerHTML = `<span>Side View: ${sideName}</span>`;

                    // Run initial inspection once
                    executeCombinedInspection();
                };
                readerSide.readAsDataURL(blobSide);
            } catch (e) {
                console.log("Side initial sample notice:", e);
            }
        }

        async function executeCombinedInspection() {
            if (!currentTopB64 || !currentSideB64) {
                alert("Please provide both Top View and Side View images (via cameras or uploads).");
                return;
            }

            const btnRun = document.getElementById('btnRunCurrent');
            btnRun.innerHTML = '<span>⏳</span> Inspecting Both Views...';
            btnRun.disabled = true;

            const confPose = 0.35;
            const confOrig3 = 0.40;
            const confSolder = 0.25;

            try {
                const resp = await fetch('/api/combined/inspect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        top_image_b64: currentTopB64,
                        side_image_b64: currentSideB64,
                        conf_pose: confPose,
                        conf_orig3: confOrig3,
                        conf_solder: confSolder
                    })
                });

                const data = await resp.json();
                if (data.success) {
                    inspectionResult = data;
                    renderInspectionResults(data);
                } else {
                    alert('Inspection failed: ' + (data.error || 'Unknown error'));
                }
            } catch (err) {
                alert('API error: ' + err);
            } finally {
                btnRun.innerHTML = '<span>⚡</span> Inspect Current Images';
                btnRun.disabled = false;
            }
        }

        function renderInspectionResults(data) {
            const overallPass = data.overall_pass;
            const posePass = data.pose_pass;
            const orig3Pass = data.orig3_pass;
            const solderPass = data.solder_pass;

            // 1. Master Verdict Card
            const card = document.getElementById('masterVerdictCard');
            card.classList.toggle('verdict-pass', overallPass);
            card.classList.toggle('verdict-fail', !overallPass);

            const icon = document.getElementById('masterVerdictIcon');
            icon.innerText = overallPass ? '✔' : '✘';

            const title = document.getElementById('masterVerdictTitle');
            title.innerText = overallPass ? 'OVERALL SYSTEM PASS: ALL CHECKS SATISFIED' : 'OVERALL DEFECT: INSPECTION CRITERIA FAILED';

            const sub = document.getElementById('masterVerdictSub');
            if (overallPass) {
                sub.innerText = `Wire Harness Pose, Side Origin 3, and Solder Ball Verification passed cleanly in ${data.total_latency_ms} ms.`;
            } else {
                sub.innerText = `Defect detected &mdash; inspect individual subsystem cards below. (${data.reasons_failed.length} failure condition(s) noted).`;
            }

            // 2. Subsystem Status Badges
            document.getElementById('txtPoseVerdict').innerText = posePass ? 'PASS' : 'FAIL';
            document.getElementById('txtPoseVerdict').style.color = posePass ? '#34d399' : '#f87171';
            document.getElementById('dotPose').className = 'badge-status-dot ' + (posePass ? 'status-dot-pass' : 'status-dot-fail');

            document.getElementById('txtOrig3Verdict').innerText = orig3Pass ? 'PASS' : 'FAIL';
            document.getElementById('txtOrig3Verdict').style.color = orig3Pass ? '#34d399' : '#f87171';
            document.getElementById('dotOrig3').className = 'badge-status-dot ' + (orig3Pass ? 'status-dot-pass' : 'status-dot-fail');

            document.getElementById('txtSolderVerdict').innerText = solderPass ? `PASS (${data.solder_count})` : `DEFECT (${data.solder_count})`;
            document.getElementById('txtSolderVerdict').style.color = solderPass ? '#34d399' : '#f87171';
            document.getElementById('dotSolder').className = 'badge-status-dot ' + (solderPass ? 'status-dot-pass' : 'status-dot-fail');

            document.getElementById('metricTotalLatency').innerText = `${data.total_latency_ms} ms`;
            document.getElementById('metricSolderCount').innerText = `${data.solder_count}`;

            // 3. Pose Card Regions
            document.getElementById('badgeCardPose').className = 'badge-pill ' + (posePass ? 'badge-pill-pass' : 'badge-pill-fail');
            document.getElementById('badgeCardPose').innerText = posePass ? 'PASS' : 'FAIL';

            const poseList = document.getElementById('poseRegionsList');
            poseList.innerHTML = '';
            (data.components || []).forEach(comp => {
                const cPass = comp.pass;
                const div = document.createElement('div');
                div.className = 'diag-item';
                div.innerHTML = `
                    <div class="diag-item-head">
                        <span>${comp.name}</span>
                        <strong style="color:${cPass ? '#34d399' : '#f87171'};">${cPass ? 'PASS' : 'FAIL'}</strong>
                    </div>
                    <div class="diag-subtext">${comp.msg || (cPass ? 'Topological wire routing correct' : 'Keypoint checks failed')}</div>
                `;
                poseList.appendChild(div);
            });

            // 4. Side Origin 3 Card
            document.getElementById('badgeCardOrig3').className = 'badge-pill ' + (orig3Pass ? 'badge-pill-pass' : 'badge-pill-fail');
            document.getElementById('badgeCardOrig3').innerText = orig3Pass ? 'PASS' : 'FAIL';
            document.getElementById('dispOrig3Status').innerText = orig3Pass ? 'DETECTED' : 'MISSING';
            document.getElementById('dispOrig3Status').style.color = orig3Pass ? '#34d399' : '#f87171';
            const center = data.origin3_details.center;
            document.getElementById('dispOrig3Coord').innerText = center ? `(${center[0]}, ${center[1]})` : 'None';
            document.getElementById('dispOrig3StatusText').innerText = orig3Pass ? 'PASS' : 'FAIL';
            document.getElementById('dispOrig3StatusText').style.color = orig3Pass ? '#34d399' : '#f87171';
            document.getElementById('dispOrig3Msg').innerText = data.origin3_details.msg;

            // 5. Solder Ball Card
            document.getElementById('badgeCardSolder').className = 'badge-pill ' + (solderPass ? 'badge-pill-pass' : 'badge-pill-fail');
            document.getElementById('badgeCardSolder').innerText = solderPass ? 'PASS' : 'DEFECT';
            document.getElementById('dispSolderCount').innerText = `${data.solder_count}`;
            document.getElementById('dispSolderQuality').innerText = solderPass ? 'PASS' : 'DEFECT';
            document.getElementById('dispSolderQuality').style.color = solderPass ? '#34d399' : '#f87171';
            document.getElementById('dispSolderCompliance').innerText = solderPass ? 'PASS' : 'DEFECT';
            document.getElementById('dispSolderCompliance').style.color = solderPass ? '#34d399' : '#f87171';
            document.getElementById('dispSolderMsg').innerText = data.solder_msg;

            // 6. Diagnostics Log Box
            const logBox = document.getElementById('diagLogBox');
            logBox.innerHTML = '';
            (data.reasons_failed || []).forEach(rf => {
                const el = document.createElement('div');
                el.className = 'diag-fail-entry';
                el.innerHTML = `<span>✘</span><span>${rf}</span>`;
                logBox.appendChild(el);
            });
            (data.passes || []).forEach(p => {
                const el = document.createElement('div');
                el.className = 'diag-pass-entry';
                el.innerHTML = `<span>✔</span><span>${p}</span>`;
                logBox.appendChild(el);
            });

            const lb = data.latency_breakdown || {};
            document.getElementById('dispLatencySummary').innerText = `Pose: ${lb.pose_ms || 0}ms | Origin 3: ${lb.orig3_ms || 0}ms | Solder: ${lb.solder_ms || 0}ms`;

            // 7. Update Displays with preferred view modes
            updateTopDisplay();
            updateSideDisplay();
        }

        function switchTopViewMode(mode) {
            activeTopMode = mode;
            ['Combined', 'Pose', 'Solder', 'Raw'].forEach(m => {
                const btn = document.getElementById(`btnModeTop${m}`);
                if (btn) btn.classList.toggle('active', m.toLowerCase() === mode);
            });
            updateTopDisplay();
        }

        function updateTopDisplay() {
            const img = document.getElementById('imgTopDisplay');
            const ph = document.getElementById('placeholderTop');
            if (!inspectionResult || !inspectionResult.images) {
                if (currentTopB64) {
                    img.src = currentTopB64;
                    img.style.display = 'block';
                    if (ph) ph.style.display = 'none';
                }
                return;
            }

            const imgs = inspectionResult.images;
            if (activeTopMode === 'combined') {
                img.src = imgs.top_combined || imgs.top_pose || currentTopB64;
            } else if (activeTopMode === 'pose') {
                img.src = imgs.top_pose || currentTopB64;
            } else if (activeTopMode === 'solder') {
                img.src = imgs.top_solder || currentTopB64;
            } else {
                img.src = imgs.top_original || currentTopB64;
            }
            img.style.display = 'block';
            if (ph) ph.style.display = 'none';
        }

        function switchSideViewMode(mode) {
            activeSideMode = mode;
            document.getElementById('btnModeSideAnnotated').classList.toggle('active', mode === 'annotated');
            document.getElementById('btnModeSideRaw').classList.toggle('active', mode === 'raw');
            updateSideDisplay();
        }

        function updateSideDisplay() {
            const img = document.getElementById('imgSideDisplay');
            const ph = document.getElementById('placeholderSide');
            if (!inspectionResult || !inspectionResult.images) {
                if (currentSideB64) {
                    img.src = currentSideB64;
                    img.style.display = 'block';
                    if (ph) ph.style.display = 'none';
                }
                return;
            }

            const imgs = inspectionResult.images;
            if (activeSideMode === 'annotated') {
                img.src = imgs.side_annotated || currentSideB64;
            } else {
                img.src = imgs.side_original || currentSideB64;
            }
            img.style.display = 'block';
            if (ph) ph.style.display = 'none';
        }

        function zoomImage(imgId, factor) {
            zoomState[imgId] = (zoomState[imgId] || 1.0) * factor;
            zoomState[imgId] = Math.max(0.5, Math.min(5.0, zoomState[imgId]));
            document.getElementById(imgId).style.transform = `scale(${zoomState[imgId]})`;
        }

        function resetZoom(imgId) {
            zoomState[imgId] = 1.0;
            document.getElementById(imgId).style.transform = 'scale(1)';
        }

        function downloadActiveTopImage() {
            const img = document.getElementById('imgTopDisplay');
            if (!img.src) return alert('No top image loaded to download.');
            const a = document.createElement('a');
            a.href = img.src;
            a.download = `top_inspection_${Date.now()}.jpg`;
            a.click();
        }

        function downloadActiveSideImage() {
            const img = document.getElementById('imgSideDisplay');
            if (!img.src) return alert('No side image loaded to download.');
            const a = document.createElement('a');
            a.href = img.src;
            a.download = `side_inspection_${Date.now()}.jpg`;
            a.click();
        }

        // Initialize on load
        window.addEventListener('DOMContentLoaded', () => {
            listCameraDevices();
            setupDragAndDrop();
            // Ready and waiting for user uploads or camera frames (no auto-inspection on startup)
        });
    </script>
</body>
</html>
"""

# ================= API ROUTES =================

@app.route("/")
def index():
    return render_template_string(COMBINED_HTML_TEMPLATE)

@app.route("/api/combined/inspect", methods=["POST"])
def api_combined_inspect():
    data = request.get_json(silent=True) or {}
    top_b64 = data.get("top_image_b64", "")
    side_b64 = data.get("side_image_b64", "")
    conf_pose = float(data.get("conf_pose", 0.35))
    conf_orig3 = float(data.get("conf_orig3", 0.40))
    conf_solder = float(data.get("conf_solder", 0.25))
    iou_solder = float(data.get("iou_solder", 0.45))
    expected_solder_count = data.get("expected_solder_count")
    if expected_solder_count is not None:
        try:
            expected_solder_count = int(expected_solder_count)
            if expected_solder_count < 0:
                expected_solder_count = None
        except Exception:
            expected_solder_count = None

    solder_tolerance = int(data.get("solder_tolerance", 0))

    top_img = decode_b64(top_b64)
    side_img = decode_b64(side_b64)

    if top_img is None or side_img is None:
        return jsonify({
            "success": False,
            "error": "Both Top View image (shared for wiring and solder balls) and Side View image (Origin 3) are required."
        }), 400

    result = run_combined_inspection(
        top_img,
        side_img,
        conf_pose=conf_pose,
        conf_orig3=conf_orig3,
        conf_solder=conf_solder,
        iou_solder=iou_solder,
        expected_solder_count=expected_solder_count,
        solder_tolerance=solder_tolerance
    )

    return jsonify(result)

@app.route("/api/combined/sample_image/<path:category>/<path:filename>")
def get_sample_image(category, filename):
    safe_filename = os.path.basename(filename)

    if category == "top":
        # Check test_samples in pose_estimation_poc
        fpath = os.path.join(POSE_DIR, "test_samples", safe_filename)
        if not os.path.exists(fpath):
            # Check solderball dataset as fallback for solder board samples
            fpath = os.path.join(SOLDER_DIR, "dataset", safe_filename)
    elif category == "side":
        # Check origin3 folder in test_samples
        fpath = os.path.join(POSE_DIR, "test_samples", "origin3", safe_filename)
    else:
        fpath = os.path.join(POSE_DIR, "test_samples", safe_filename)

    if os.path.exists(fpath):
        return send_file(fpath)

    return f"Sample image {safe_filename} not found in category {category}", 404

def get_models_status():
    return {
        "pose_loaded": get_pose_model() is not None,
        "origin3_loaded": get_origin3_model() is not None,
        "solder_loaded": solder_app.get_or_load_model() is not None
    }

@app.route("/api/combined/status")
def api_status():
    return jsonify(get_models_status())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
