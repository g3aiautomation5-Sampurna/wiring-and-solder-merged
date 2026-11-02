"""
Wire Harness Pose Estimation, Inspection, Annotation & Retraining Platform
Single-page Browser-Based Application with:
1. Inspection Station (Live Camera + Upload with neatly arranged component cards)
2. Interactive Keypoint Annotation Studio (Add/Remove custom wire classes, 3-point keypoint clicker)
3. Model Retraining Engine (Live epoch tracking, logs, and hot-reloading)
"""

import os
import sys
import io
import time
import json
import base64
import glob
import shutil
import argparse
import webbrowser
import threading
from collections import deque
import cv2
import numpy as np
import yaml
from flask import Flask, render_template_string, request, jsonify, send_file
from ultralytics import YOLO

# Import inspection engine
from inspect_pose import inspect_image
from dual_inspection import inspect_dual_system, inspect_origin3, get_origin3_model, get_pose_model, reload_origin3_model

app = Flask(__name__)

# Global model instance
MODEL = None
MODEL_NAME = "best.pt"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML_PATH = os.path.join(SCRIPT_DIR, "data.yaml")
COLORS_PATH = os.path.join(SCRIPT_DIR, "class_colors.json")

# Origin 3 Side Model dataset paths
ORIGIN3_DATASET_DIR = os.path.join(SCRIPT_DIR, "origin3_dataset")
ORIGIN3_IMAGES_DIR = os.path.join(ORIGIN3_DATASET_DIR, "images")
ORIGIN3_LABELS_DIR = os.path.join(ORIGIN3_DATASET_DIR, "labels")
ORIGIN3_YOLO_DIR = os.path.join(ORIGIN3_DATASET_DIR, "yolo_dataset")
ORIGIN3_YAML_PATH = os.path.join(ORIGIN3_YOLO_DIR, "data.yaml")
os.makedirs(ORIGIN3_IMAGES_DIR, exist_ok=True)
os.makedirs(ORIGIN3_LABELS_DIR, exist_ok=True)

# Default color palette for wire classes
DEFAULT_PALETTE = [
    "#ef4444", # Red (Class 0)
    "#334155", # Black/Dark Slate (Class 1)
    "#3b82f6", # Blue (Class 2)
    "#eab308", # Yellow (Class 3)
    "#10b981", # Green (Class 4)
    "#f8fafc", # White (Class 5)
    "#f97316", # Orange (Class 6)
    "#a855f7", # Purple (Class 7)
    "#ec4899", # Pink (Class 8)
    "#06b6d4"  # Cyan (Class 9)
]

def get_model():
    global MODEL, MODEL_NAME
    if MODEL is None:
        weights_path = os.path.join(SCRIPT_DIR, "runs", "wire_pose_model", "weights", "best.pt")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(SCRIPT_DIR, "yolov8n-pose.pt")
        MODEL_NAME = os.path.basename(weights_path)
        print(f"Loading YOLO-Pose model from: {weights_path}")
        MODEL = YOLO(weights_path)
    return MODEL

def reload_model_after_training():
    global MODEL, MODEL_NAME
    weights_path = os.path.join(SCRIPT_DIR, "runs", "wire_pose_model", "weights", "best.pt")
    if os.path.exists(weights_path):
        MODEL = YOLO(weights_path)
        MODEL_NAME = "best.pt (Retrained)"
        print(f"Model successfully hot-reloaded: {weights_path}")

def cv2_to_base64(bgr_img):
    if bgr_img is None:
        return ""
    success, encoded = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("utf-8")

def get_classes_data():
    if not os.path.exists(DATA_YAML_PATH):
        return []
    with open(DATA_YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    
    color_map = {}
    if os.path.exists(COLORS_PATH):
        try:
            with open(COLORS_PATH, "r") as cf:
                color_map = json.load(cf)
        except Exception:
            pass

    classes_list = []
    for cid, cname in names.items():
        cid_int = int(cid)
        color = color_map.get(str(cid_int), DEFAULT_PALETTE[cid_int % len(DEFAULT_PALETTE)])
        classes_list.append({"id": cid_int, "name": str(cname), "color": color})
    classes_list.sort(key=lambda x: x["id"])
    return classes_list

def save_class_color(class_id, color):
    color_map = {}
    if os.path.exists(COLORS_PATH):
        try:
            with open(COLORS_PATH, "r") as cf:
                color_map = json.load(cf)
        except Exception:
            pass
    color_map[str(class_id)] = color
    with open(COLORS_PATH, "w") as cf:
        json.dump(color_map, cf, indent=2)

# ----------------- TRAINING MANAGER -----------------
class TrainingManager:
    def __init__(self):
        self.status = "idle" # idle, training, completed, error
        self.total_epochs = 0
        self.current_epoch = 0
        self.progress_percent = 0
        self.logs = []
        self.error_msg = ""
        self.thread = None

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        print(f"[TRAIN] {msg}")
        self.logs.append(entry)
        if len(self.logs) > 300:
            self.logs = self.logs[-300:]

    def start(self, epochs=20, batch=8, weights_choice="best"):
        if self.status == "training":
            return False, "Training is already in progress."

        self.status = "training"
        self.total_epochs = epochs
        self.current_epoch = 0
        self.progress_percent = 0
        self.logs = []
        self.error_msg = ""

        self.thread = threading.Thread(target=self._run_train, args=(epochs, batch, weights_choice), daemon=True)
        self.thread.start()
        return True, "Training started in background."

    def _run_train(self, epochs, batch, weights_choice):
        try:
            self.log(f"Starting YOLOv8-Pose retraining: {epochs} epochs, batch size {batch}...")
            
            # Clear label caches
            for c_file in glob.glob(os.path.join(SCRIPT_DIR, "dataset", "labels", "*.cache")):
                try: os.remove(c_file)
                except Exception: pass

            best_weights = os.path.join(SCRIPT_DIR, "runs", "wire_pose_model", "weights", "best.pt")
            if weights_choice == "best" and os.path.exists(best_weights):
                base_model_path = best_weights
                self.log(f"Fine-tuning from existing best model: {os.path.basename(best_weights)}")
            else:
                base_model_path = os.path.join(SCRIPT_DIR, "yolov8n-pose.pt")
                self.log(f"Training from base nano model: {os.path.basename(base_model_path)}")

            model = YOLO(base_model_path)

            def on_epoch_end(trainer):
                try:
                    ep = trainer.epoch + 1
                    total = trainer.epochs
                    self.current_epoch = ep
                    self.progress_percent = int((ep / total) * 100)
                    loss_str = ""
                    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                        items = trainer.loss_items
                        if isinstance(items, dict):
                            total_loss = sum(float(v) for v in items.values())
                            loss_str = f" | Loss: {total_loss:.4f}"
                        elif hasattr(items, '__getitem__'):
                            loss_str = f" | Loss: {float(items[0]):.4f}"
                    self.log(f"Epoch {ep}/{total} finished [{self.progress_percent}%]{loss_str}")
                except Exception as cb_err:
                    print(f"[CALLBACK WARN] {cb_err}")

            model.add_callback("on_train_epoch_end", on_epoch_end)

            self.log("Training engine running...")
            results = model.train(
                data=DATA_YAML_PATH,
                epochs=epochs,
                imgsz=640,
                batch=batch,
                device="cpu",
                workers=0,
                fliplr=0.0,
                project=os.path.join(SCRIPT_DIR, "runs"),
                name="wire_pose_model",
                exist_ok=True,
                plots=True,
                verbose=False
            )

            self.status = "completed"
            self.progress_percent = 100
            self.log("Training successfully finished! New best weights generated.")
            reload_model_after_training()
            self.log("Live inspection engine updated with the newly retrained model weights.")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[TRAIN ERROR]\n{tb}")
            self.status = "error"
            self.error_msg = str(e)
            self.log(f"Training error encountered: {type(e).__name__}: {str(e)}")

training_mgr = TrainingManager()


def prepare_origin3_yolo_dataset(split_ratio=0.8, seed=42):
    import random
    random.seed(seed)

    all_images = sorted(
        glob.glob(os.path.join(ORIGIN3_IMAGES_DIR, "*.jpg")) +
        glob.glob(os.path.join(ORIGIN3_IMAGES_DIR, "*.png")) +
        glob.glob(os.path.join(ORIGIN3_IMAGES_DIR, "*.jpeg"))
    )

    if not all_images:
        return False, "No images found in origin3_dataset/images"

    labeled_pairs = []
    unlabeled_images = []

    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(ORIGIN3_LABELS_DIR, f"{base_name}.txt")
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            labeled_pairs.append((img_path, lbl_path))
        else:
            unlabeled_images.append(img_path)

    if not labeled_pairs:
        return False, "No labeled images found in origin3_dataset/labels. Please annotate at least one image."

    random.shuffle(labeled_pairs)
    train_count = max(1, int(len(labeled_pairs) * split_ratio))
    train_pairs = labeled_pairs[:train_count]
    val_pairs = labeled_pairs[train_count:]
    if not val_pairs:
        val_pairs = [train_pairs[-1]]

    dirs = {
        "train_img": os.path.join(ORIGIN3_YOLO_DIR, "images", "train"),
        "val_img": os.path.join(ORIGIN3_YOLO_DIR, "images", "val"),
        "train_lbl": os.path.join(ORIGIN3_YOLO_DIR, "labels", "train"),
        "val_lbl": os.path.join(ORIGIN3_YOLO_DIR, "labels", "val"),
    }

    if os.path.exists(ORIGIN3_YOLO_DIR):
        try:
            shutil.rmtree(ORIGIN3_YOLO_DIR)
        except Exception:
            pass
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    for img, lbl in train_pairs:
        shutil.copy2(img, dirs["train_img"])
        shutil.copy2(lbl, dirs["train_lbl"])

    for img, lbl in val_pairs:
        shutil.copy2(img, dirs["val_img"])
        shutil.copy2(lbl, dirs["val_lbl"])

    yaml_content = {
        "path": ORIGIN3_YOLO_DIR.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "origin_3"
        }
    }
    with open(ORIGIN3_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    return True, f"Origin 3 dataset split complete: {len(train_pairs)} train, {len(val_pairs)} val images."


class Origin3TrainingManager:
    def __init__(self):
        self.status = "idle"
        self.total_epochs = 0
        self.current_epoch = 0
        self.progress_percent = 0
        self.logs = []
        self.error_msg = ""
        self.thread = None

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        print(f"[ORIGIN3-TRAIN] {msg}")
        self.logs.append(entry)
        if len(self.logs) > 300:
            self.logs = self.logs[-300:]

    def start(self, epochs=30, batch=8, weights_choice="best"):
        if self.status == "training":
            return False, "Training is already in progress for Origin 3."

        self.status = "training"
        self.total_epochs = epochs
        self.current_epoch = 0
        self.progress_percent = 0
        self.logs = []
        self.error_msg = ""

        self.thread = threading.Thread(target=self._run_train, args=(epochs, batch, weights_choice), daemon=True)
        self.thread.start()
        return True, "Origin 3 retraining started in background."

    def _run_train(self, epochs, batch, weights_choice):
        try:
            self.log(f"Preparing Origin 3 YOLO dataset split (80/20)...")
            ok, prep_msg = prepare_origin3_yolo_dataset()
            if not ok:
                raise RuntimeError(prep_msg)
            self.log(prep_msg)

            # Clear cache files
            for pattern in [
                os.path.join(ORIGIN3_YOLO_DIR, "labels", "*.cache"),
                os.path.join(ORIGIN3_YOLO_DIR, "labels", "*", "*.cache")
            ]:
                for c_file in glob.glob(pattern):
                    try: os.remove(c_file)
                    except Exception: pass

            best_weights = os.path.join(SCRIPT_DIR, "runs", "origin3_model", "weights", "best.pt")
            if not os.path.exists(best_weights):
                best_weights = os.path.join(os.path.dirname(SCRIPT_DIR), "runs", "origin3_model", "weights", "best.pt")

            if weights_choice == "best" and os.path.exists(best_weights):
                base_model_path = best_weights
                self.log(f"Fine-tuning from existing best model: {os.path.basename(best_weights)}")
            else:
                base_model_path = os.path.join(SCRIPT_DIR, "yolov8n.pt")
                if not os.path.exists(base_model_path):
                    base_model_path = os.path.join(os.path.dirname(SCRIPT_DIR), "yolov8n.pt")
                if not os.path.exists(base_model_path):
                    base_model_path = "yolov8n.pt"
                self.log(f"Training from base nano weights: {os.path.basename(base_model_path)}")

            self.log(f"Loading YOLO model: {base_model_path}")
            model = YOLO(base_model_path)

            def on_epoch_end(trainer):
                try:
                    ep = trainer.epoch + 1
                    total = trainer.epochs
                    self.current_epoch = ep
                    self.progress_percent = int((ep / total) * 100)
                    loss_str = ""
                    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                        items = trainer.loss_items
                        if isinstance(items, dict):
                            total_loss = sum(float(v) for v in items.values())
                            loss_str = f" | Loss: {total_loss:.4f}"
                        elif hasattr(items, '__getitem__'):
                            loss_str = f" | Loss: {float(items[0]):.4f}"
                    self.log(f"Epoch {ep}/{total} finished [{self.progress_percent}%]{loss_str}")
                except Exception as cb_err:
                    print(f"[CALLBACK WARN] {cb_err}")

            model.add_callback("on_train_epoch_end", on_epoch_end)

            self.log(f"Starting Ultralytics YOLOv8 training: {epochs} epochs, batch size {batch}...")
            results = model.train(
                data=ORIGIN3_YAML_PATH,
                epochs=epochs,
                imgsz=640,
                batch=batch,
                device="cpu",
                workers=0,
                project=os.path.join(SCRIPT_DIR, "runs"),
                name="origin3_model",
                exist_ok=True,
                plots=True,
                verbose=False
            )

            # Sync generated weights to workspace runs directory as well
            gen_best = os.path.join(SCRIPT_DIR, "runs", "origin3_model", "weights", "best.pt")
            ws_runs_best = os.path.join(os.path.dirname(SCRIPT_DIR), "runs", "origin3_model", "weights", "best.pt")
            if os.path.exists(gen_best):
                try:
                    os.makedirs(os.path.dirname(ws_runs_best), exist_ok=True)
                    shutil.copy2(gen_best, ws_runs_best)
                except Exception as sync_err:
                    self.log(f"[WARN] Sync weights error: {sync_err}")

            self.status = "completed"
            self.progress_percent = 100
            self.log("Origin 3 training successfully completed! Generated new best.pt.")
            reload_origin3_model()
            self.log("Side Inspection model hot-reloaded with updated weights in dual inspection engine.")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ORIGIN3 TRAIN ERROR]\n{tb}")
            self.status = "error"
            self.error_msg = str(e)
            self.log(f"Origin 3 training error: {type(e).__name__}: {str(e)}")


origin3_training_mgr = Origin3TrainingManager()

def parse_dual_result(dual_res):
    pose_res = dual_res.get("pose_result", {})
    orig3_res = dual_res.get("origin3_result", {})
    regions = pose_res.get("regions", {})
    components = []

    def make_component_data(comp_id, default_name):
        reg = regions.get(comp_id, {})
        c_pass = reg.get("pass", False)
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
            "name": term_data.get("name", "Wire Terminal / Socket"),
            "pass": term_data.get("pass", False),
            "rule": term_data.get("rule", "Terminal Position Check"),
            "detail": term_data.get("detail", "Not evaluated")
        })

        return {
            "id": comp_id,
            "name": reg.get("name", default_name),
            "pass": c_pass,
            "msg": reg.get("msg", ""),
            "checks": checks,
            "red_detected": len(reg.get("red", [])),
            "black_detected": len(reg.get("black", []))
        }

    components.append(make_component_data("UM", "Upper Middle (Coils)"))
    components.append(make_component_data("BL", "Below Left (Terminals)"))
    components.append(make_component_data("BR", "Below Right (Exit / Grommet)"))

    pose_annotated_b64 = cv2_to_base64(pose_res.get("annotated_img"))
    pose_orig_b64 = cv2_to_base64(pose_res.get("orig_img"))
    orig3_annotated_b64 = cv2_to_base64(orig3_res.get("annotated_img"))
    orig3_orig_b64 = cv2_to_base64(orig3_res.get("orig_img"))

    return {
        "success": True,
        "overall_pass": bool(dual_res.get("overall_pass", False)),
        "is_pass": bool(dual_res.get("overall_pass", False)),
        "pose_pass": bool(dual_res.get("pose_pass", False)),
        "orig3_pass": bool(dual_res.get("orig3_pass", False)),
        "latency_ms": dual_res.get("total_latency_ms", 0.0),
        "pose_latency_ms": round(pose_res.get("latency_ms", 0.0), 1),
        "orig3_latency_ms": round(orig3_res.get("latency_ms", 0.0), 1),
        "total_wires": pose_res.get("total_wires", 0),
        # Primary images for top view
        "annotated_image": pose_annotated_b64,
        "original_image": pose_orig_b64,
        "annotated_image_top": pose_annotated_b64,
        "original_image_top": pose_orig_b64,
        # Secondary images for Origin 3 side view
        "annotated_image_orig3": orig3_annotated_b64,
        "original_image_orig3": orig3_orig_b64,
        # Origin 3 card data
        "origin3_card": {
            "is_pass": bool(orig3_res.get("is_pass", False)),
            "confidence": round(orig3_res.get("confidence", 0.0) * 100, 1),
            "center": orig3_res.get("center"),
            "bbox": orig3_res.get("bbox"),
            "msg": orig3_res.get("msg", "")
        },
        "components": components,
        "reasons_failed": dual_res.get("reasons_failed", []),
        "passes": dual_res.get("passes", []),
        "model_name": "YOLOv8-Pose (Top) + YOLOv8-Origin3 (Side)"
    }

def parse_inspection_result(res):
    # Backward compatible wrapper if only single pose result given
    dummy_orig3 = {
        "is_pass": True,
        "confidence": 1.0,
        "center": [0, 0],
        "bbox": [0, 0, 0, 0],
        "msg": "Origin 3 (Not evaluated in single mode)",
        "annotated_img": None,
        "orig_img": None,
        "latency_ms": 0.0
    }
    dummy_dual = {
        "overall_pass": res.get("is_pass", False),
        "pose_pass": res.get("is_pass", False),
        "orig3_pass": True,
        "total_latency_ms": res.get("latency_ms", 0.0),
        "reasons_failed": res.get("reasons_failed", []),
        "passes": res.get("passes", []),
        "pose_result": res,
        "origin3_result": dummy_orig3
    }
    return parse_dual_result(dummy_dual)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wire Harness Pose Inspection & Annotation Platform</title>
    <style>
        :root {
            --bg-main: #0b0f17;
            --bg-panel: #131b26;
            --bg-card: #1c2738;
            --bg-card-hover: #223046;
            --border: #2a3a52;
            --border-accent: #3b82f6;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --pass-bg: #064e3b;
            --pass-border: #10b981;
            --pass-text: #34d399;
            --fail-bg: #7f1d1d;
            --fail-border: #ef4444;
            --fail-text: #f87171;
            --blue: #2563eb;
            --blue-hover: #1d4ed8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: var(--bg-main);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Top Header */
        header {
            background-color: var(--bg-panel);
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-title h1 {
            font-size: 1.12rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .nav-tabs {
            display: flex;
            gap: 6px;
            background: var(--bg-card);
            padding: 4px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .nav-tab-btn {
            background: transparent;
            color: var(--text-muted);
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .nav-tab-btn.active {
            background: var(--blue);
            color: #ffffff;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.4);
        }

        .header-badges {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .badge-model {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .badge-live {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .slider-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
            color: var(--text-muted);
            background: var(--bg-card);
            padding: 4px 12px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }

        .slider-wrap input[type=range] {
            accent-color: var(--blue);
            cursor: pointer;
            width: 80px;
        }

        /* Views Container */
        .main-view {
            flex: 1;
            display: none;
            padding: 10px 14px;
            box-sizing: border-box;
            max-width: 100%;
            overflow-x: hidden;
        }

        .main-view.active {
            display: flex;
        }

        /* ================= 1. INSPECTION VIEW ================= */
        .workspace {
            display: grid;
            grid-template-columns: 1fr 420px;
            flex: 1;
            gap: 16px;
            width: 100%;
        }

        @media (max-width: 1080px) {
            .workspace {
                grid-template-columns: 1fr;
            }
        }

        .stage-panel {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .mode-bar {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }

        .tab-group {
            display: flex;
            gap: 6px;
        }

        .tab-btn {
            background: transparent;
            color: var(--text-muted);
            border: none;
            padding: 7px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.84rem;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .tab-btn.active {
            background: var(--blue);
            color: #ffffff;
        }

        select {
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.82rem;
            outline: none;
            cursor: pointer;
        }

        .stage-box {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
            min-height: 480px;
        }

        .stage-header {
            padding: 10px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(19, 27, 38, 0.7);
        }

        .view-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.76rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
        }

        .view-btn.active {
            background: #334155;
            color: #ffffff;
            border-color: #64748b;
        }

        .display-area {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            background: #06090e;
            overflow: hidden;
        }

        video, #displayImg {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: block;
        }

        .camera-action-bar {
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(15, 23, 42, 0.88);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 8px 18px;
            border-radius: 30px;
            display: flex;
            align-items: center;
            gap: 12px;
            z-index: 10;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
        }

        .btn-snap {
            background: var(--blue);
            color: white;
            border: none;
            padding: 9px 20px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        }

        .btn-snap:hover {
            background: var(--blue-hover);
        }

        .upload-dropzone {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 32px;
            border: 2px dashed var(--border);
            border-radius: 8px;
            margin: 20px;
            cursor: pointer;
            transition: all 0.2s;
            background: rgba(28, 39, 56, 0.4);
            text-align: center;
        }

        .upload-dropzone:hover {
            border-color: var(--blue);
            background: rgba(37, 99, 235, 0.08);
        }

        /* Dual Camera and Dual Upload Layout */
        .dual-stream-container, .dual-upload-container, .dual-result-container {
            display: flex;
            gap: 12px;
            width: 100%;
            height: 100%;
            padding: 12px;
            box-sizing: border-box;
        }

        .stream-box, .upload-half, .result-half {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            background: #06090e;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            min-height: 380px;
        }

        .stream-tag {
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 10;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(4px);
            color: #f1f5f9;
            font-size: 0.74rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            letter-spacing: 0.5px;
        }

        .upload-half {
            cursor: pointer;
            background: var(--bg-card);
            transition: all 0.2s;
        }

        .upload-half:hover {
            border-color: var(--blue);
            background: var(--bg-card-hover);
        }

        .upload-half .upload-dropzone {
            margin: 0;
            width: 100%;
            height: 100%;
            border: 2px dashed var(--border);
            border-radius: 8px;
            padding: 20px;
            box-sizing: border-box;
            background: transparent;
        }

        .upload-preview {
            max-width: 90%;
            max-height: 240px;
            object-fit: contain;
            margin-top: 12px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }

        .result-display-img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .sub-verdict-pill {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.74rem;
            font-weight: 700;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #cbd5e1;
        }

        .sub-verdict-pill.pass {
            background: var(--pass-bg);
            color: var(--pass-text);
            border-color: var(--pass-border);
        }

        .sub-verdict-pill.fail {
            background: var(--fail-bg);
            color: var(--fail-text);
            border-color: var(--fail-border);
        }

        .btn-start {
            background: #10b981;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 24px;
            font-weight: 800;
            font-size: 0.95rem;
            letter-spacing: 0.5px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
        }

        .btn-start:hover {
            background: #059669;
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(16, 185, 129, 0.5);
        }

        .btn-start:active {
            transform: translateY(1px);
        }

        .btn-retake {
            background: #334155;
            color: white;
            border: 1px solid var(--border);
            padding: 8px 18px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        }

        .btn-retake:hover {
            background: #475569;
        }

        .shutter-flash {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: white;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.15s ease-out;
            z-index: 25;
        }

        .shutter-flash.flash {
            opacity: 0.75;
        }

        /* Fullscreen Lightbox Modal */
        .fs-modal {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .fs-modal.hidden {
            display: none !important;
        }

        .fs-modal-backdrop {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(4, 7, 13, 0.94);
            backdrop-filter: blur(8px);
        }

        .fs-modal-content {
            position: relative;
            width: 97vw;
            height: 95vh;
            background: #0b0f17;
            border: 1px solid var(--border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.85);
            z-index: 10;
        }

        .fs-modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 20px;
            background: #131b26;
            border-bottom: 1px solid var(--border);
        }

        .fs-header-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .fs-header-tabs {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .fs-close-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: white;
            font-size: 1.1rem;
            font-weight: bold;
            width: 34px;
            height: 34px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s;
        }

        .fs-close-btn:hover {
            background: #ef4444;
            border-color: #ef4444;
        }

        .fs-image-body {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #03060a;
            overflow: hidden;
            position: relative;
            padding: 12px;
        }

        .fs-dual-view {
            display: flex;
            gap: 16px;
            width: 100%;
            height: 100%;
        }

        .fs-half {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000000;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .fs-half img, .fs-single-view img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: block;
        }

        .fs-single-view {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .fs-modal-footer {
            padding: 8px 20px;
            background: #131b26;
            border-top: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .sample-pills-bar {
            padding: 8px 14px;
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .sample-pill {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 5px 12px;
            border-radius: 16px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .sample-pill:hover {
            border-color: #60a5fa;
            background: var(--bg-card-hover);
        }

        .sample-pill.pass { border-left: 3px solid #10b981; }
        .sample-pill.fail { border-left: 3px solid #ef4444; }

        /* Right Panel Cards */
        .results-panel {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .verdict-card {
            border-radius: 10px;
            padding: 18px;
            text-align: center;
            border: 2px solid var(--border);
            background: var(--bg-panel);
            transition: all 0.3s ease;
        }

        .verdict-card.ready { background: var(--bg-card); color: var(--text-muted); }
        .verdict-card.pass {
            background: linear-gradient(145deg, #064e3b 0%, #022c22 100%);
            border-color: #10b981;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
        }
        .verdict-card.fail {
            background: linear-gradient(145deg, #7f1d1d 0%, #450a0a 100%);
            border-color: #ef4444;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
        }

        .verdict-title {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: 1px;
        }
        .verdict-card.pass .verdict-title { color: #34d399; }
        .verdict-card.fail .verdict-title { color: #f87171; }
        .verdict-card.ready .verdict-title { color: #cbd5e1; }

        .metrics-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            margin-top: 8px;
            font-size: 0.8rem;
        }
        .metric-item {
            background: rgba(0, 0, 0, 0.25);
            padding: 4px 10px;
            border-radius: 4px;
        }

        .component-card {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .component-card.pass { border-left: 4px solid #10b981; }
        .component-card.fail { border-left: 4px solid #ef4444; }

        .comp-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .comp-name { font-size: 0.9rem; font-weight: 700; color: #ffffff; }
        .comp-status-badge {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
        }
        .comp-status-badge.pass { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
        .comp-status-badge.fail { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }

        .check-item {
            background: rgba(11, 15, 23, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            padding: 6px 10px;
            display: flex;
            flex-direction: column;
            gap: 2px;
            font-size: 0.76rem;
        }
        .check-top { display: flex; align-items: center; justify-content: space-between; }
        .check-label { font-weight: 600; color: #cbd5e1; }
        .check-pill { font-size: 0.68rem; font-weight: 700; padding: 1px 6px; border-radius: 3px; }
        .check-pill.pass { color: #34d399; background: rgba(16, 185, 129, 0.15); }
        .check-pill.fail { color: #f87171; background: rgba(239, 68, 68, 0.15); }
        .check-rule { color: var(--text-muted); font-size: 0.7rem; }
        .check-detail { color: #e2e8f0; font-family: Consolas, monospace; font-size: 0.73rem; }

        .defects-box {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
            padding: 12px;
            font-size: 0.8rem;
        }
        .defects-box h4 { color: #f87171; margin-bottom: 6px; }
        .defects-box ul { padding-left: 20px; color: #fca5a5; }

        /* ================= 2. ANNOTATION & RETRAINING VIEW ================= */
        .filter-tabs-row {
            display: flex;
            gap: 4px;
            margin-top: 2px;
            margin-bottom: 4px;
        }

        .filter-pill {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.7rem;
            padding: 3px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.15s;
        }

        .filter-pill:hover {
            color: #ffffff;
            border-color: #64748b;
        }

        .filter-pill.active {
            background: rgba(6, 182, 212, 0.2);
            color: #38bdf8;
            border-color: #0284c7;
        }

        .studio-workspace {
            display: grid;
            grid-template-columns: 240px minmax(0, 1fr) 280px;
            gap: 12px;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            align-items: start;
        }

        @media (max-width: 1100px) {
            .studio-workspace {
                grid-template-columns: 1fr;
            }
        }

        .studio-card {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            min-width: 0;
            max-width: 100%;
            overflow: hidden;
            box-sizing: border-box;
        }

        .card-header-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Class Manager Chips */
        .class-chips-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }

        .class-chip {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 4px 10px;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-main);
            display: inline-flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .class-chip.active {
            border-color: #ffffff;
            background: var(--bg-card-hover);
            box-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
        }

        .color-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }

        .chip-del-btn {
            color: #94a3b8;
            margin-left: 4px;
            font-weight: bold;
            cursor: pointer;
        }
        .chip-del-btn:hover { color: #ef4444; }

        .btn-add-class {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px dashed rgba(59, 130, 246, 0.4);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.76rem;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-add-class:hover { background: rgba(59, 130, 246, 0.25); }

        /* Image Explorer list */
        .image-list-box {
            max-height: 380px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .image-item-row {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 10px;
            font-size: 0.78rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            transition: all 0.2s;
        }
        .image-item-row:hover { background: var(--bg-card-hover); border-color: #64748b; }
        .image-item-row.active { border-color: var(--blue); background: rgba(37, 99, 235, 0.15); }

        /* Interactive Canvas */
        .canvas-container {
            background: #06090e;
            border: 1px solid var(--border);
            border-radius: 8px;
            position: relative;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            min-height: 280px;
            max-height: calc(100vh - 230px);
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
        }

        #annoCanvas, #sideCanvas {
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
            object-fit: contain;
            cursor: crosshair;
            display: block;
        }

        .canvas-toolbar {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 6px;
            flex-shrink: 0;
        }

        .tool-btn {
            background: #273549;
            color: #ffffff;
            border: 1px solid var(--border);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.76rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            white-space: nowrap;
        }
        .tool-btn:hover { background: #33445e; }
        .tool-btn.primary { background: var(--blue); border-color: var(--blue); }
        .tool-btn.primary:hover { background: var(--blue-hover); }

        .wire-item-pill {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            padding: 6px 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.76rem;
            margin-bottom: 4px;
        }

        /* Class Selector Toolbar */
        .anno-class-selector-bar {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #0d1522;
            border: 1px solid #334155;
            padding: 4px 10px;
            border-radius: 6px;
            flex-wrap: wrap;
        }

        .anno-class-select {
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #475569;
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            outline: none;
        }
        .anno-class-select:focus {
            border-color: #3b82f6;
        }

        .anno-class-tick-group {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }

        .anno-class-tick-label {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-size: 0.74rem;
            font-weight: 600;
            cursor: pointer;
            padding: 2px 7px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.1);
            user-select: none;
            transition: all 0.15s;
        }
        .anno-class-tick-label:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .anno-class-tick-label.active {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.2);
            box-shadow: 0 0 6px rgba(59, 130, 246, 0.3);
        }
        .anno-class-tick-input {
            cursor: pointer;
            margin: 0;
            accent-color: #3b82f6;
        }

        .finished-wires-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 10px;
            flex-shrink: 0;
            box-sizing: border-box;
            transition: all 0.2s ease;
        }

        .finished-wires-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .btn-toggle-wires {
            background: transparent;
            border: 1px solid #334155;
            color: #94a3b8;
            border-radius: 4px;
            padding: 1px 8px;
            font-size: 0.7rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn-toggle-wires:hover {
            color: #f8fafc;
            border-color: #64748b;
            background: rgba(255, 255, 255, 0.05);
        }

        .finished-wires-body {
            max-height: 85px;
            overflow-y: auto;
            margin-top: 4px;
        }

        .finished-wires-box.collapsed .finished-wires-body {
            display: none !important;
        }

        /* Fullscreen Annotation Mode */
        .canvas-card-fullscreen {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 9999 !important;
            background: #020617 !important;
            padding: 8px 12px !important;
            margin: 0 !important;
            border-radius: 0 !important;
            border: none !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 6px !important;
            box-sizing: border-box !important;
        }

        .canvas-card-fullscreen .canvas-toolbar {
            position: relative !important;
            top: auto !important;
            left: auto !important;
            right: auto !important;
            flex-shrink: 0 !important;
            width: 100% !important;
            background: #0f172a !important;
            border: 1px solid #1e293b !important;
            border-radius: 8px !important;
            padding: 6px 12px !important;
            box-sizing: border-box !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        }

        .canvas-card-fullscreen .canvas-container {
            flex: 1 1 0 !important;
            min-height: 0 !important;
            min-width: 0 !important;
            width: 100% !important;
            height: auto !important;
            border: 1px solid #1e293b !important;
            border-radius: 8px !important;
            background: #020617 !important;
            padding: 0 !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
        }

        .canvas-card-fullscreen .canvas-container #annoCanvas,
        .canvas-card-fullscreen .canvas-container #sideCanvas {
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            display: block !important;
        }

        .canvas-card-fullscreen .finished-wires-box {
            position: relative !important;
            bottom: auto !important;
            left: auto !important;
            right: auto !important;
            flex-shrink: 0 !important;
            width: 100% !important;
            background: #0f172a !important;
            border: 1px solid #1e293b !important;
            border-radius: 8px !important;
            padding: 5px 10px !important;
            box-sizing: border-box !important;
            max-height: 90px !important;
        }

        /* Retrain Panel Console */
        .retrain-console {
            background: #020617;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 10px;
            font-family: Consolas, monospace;
            font-size: 0.72rem;
            color: #38bdf8;
            max-height: 220px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        .progress-bar-wrap {
            background: #1e293b;
            border-radius: 6px;
            height: 10px;
            overflow: hidden;
            margin: 6px 0;
        }

        .progress-bar-fill {
            background: linear-gradient(90deg, #3b82f6, #10b981);
            height: 100%;
            width: 0%;
            transition: width 0.3s;
        }

        /* Modal Overlay */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(4px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }

        .modal-box {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            width: 360px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        input[type=text], input[type=number] {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            outline: none;
        }
        input[type=text]:focus, input[type=number]:focus { border-color: var(--blue); }

        .hidden { display: none !important; }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="header-title">
            <h1>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2">
                    <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
                </svg>
                WIRE HARNESS POSE INSPECTION
            </h1>

            <!-- Top View Switcher -->
            <div class="nav-tabs">
                <button class="nav-tab-btn active" id="btnNavInspect" onclick="switchMainTab('inspect')" title="Live Dual Camera Inspection Station">
                    🔍 Inspection Station
                </button>
                <button class="nav-tab-btn" id="btnNavTopStudio" onclick="switchMainTab('top_studio')" title="Top View Wire Pose Keypoint Annotation & Retraining">
                    📐 Top Annotation & Retraining
                </button>
                <button class="nav-tab-btn" id="btnNavSideStudio" onclick="switchMainTab('side_studio')" title="Side View Origin 3 Bounding Box Annotation & Retraining">
                    🔌 Side Annotation & Retraining
                </button>
            </div>
        </div>

        <div class="header-badges">
            <span class="badge badge-model">Model: <span id="modelNameText">{{ model_name }}</span></span>
            <div class="slider-wrap">
                <span>Confidence:</span>
                <input type="range" id="confSlider" min="10" max="80" value="35" step="5">
                <span id="confVal" style="font-weight: bold; color: #60a5fa;">0.35</span>
            </div>
            <span class="badge badge-live">● System Ready</span>
        </div>
    </header>

    <!-- ================= VIEW 1: INSPECTION STATION ================= -->
    <div class="main-view active" id="viewInspect">
        <div class="workspace">
            <!-- Left Stage -->
            <div class="stage-panel">
                <div class="mode-bar">
                    <div class="tab-group">
                        <button class="tab-btn active" id="tabCamera" onclick="switchInspectMode('camera')">
                            📷 Live Dual Cameras
                        </button>
                        <button class="tab-btn" id="tabUpload" onclick="switchInspectMode('upload')">
                            📁 Upload Two Images
                        </button>
                    </div>
                    <div class="camera-controls" id="cameraSelectWrap" style="display:flex; gap:10px; align-items:center;">
                        <span style="font-size:0.75rem; color:#94a3b8;">Cam 1 (Top Pose):</span>
                        <select id="cameraSelect1" onchange="startCameras()"></select>
                        <span style="font-size:0.75rem; color:#94a3b8; margin-left:6px;">Cam 2 (Origin 3):</span>
                        <select id="cameraSelect2" onchange="startCameras()"></select>
                    </div>
                </div>

                <!-- Quick Load Sample Pills -->
                <div class="sample-pills-bar hidden" id="samplesBar">
                    <span style="font-size: 0.74rem; font-weight: 700; color: #94a3b8;">DUAL TEST PAIRS:</span>
                    <button class="sample-pill pass" onclick="loadSamplePair('pass')">✔ Both Pass Pair</button>
                    <button class="sample-pill fail" onclick="loadSamplePair('fail_pose')">✘ Top Pose Fail (BL)</button>
                    <button class="sample-pill fail" onclick="loadSamplePair('fail_orig3')">✘ Origin 3 Missing</button>
                </div>

                <!-- Display Stage Box -->
                <div class="stage-box">
                    <div class="stage-header">
                        <span id="stageTitle" style="font-size: 0.85rem; font-weight: 700; color: #cbd5e1;">Live Dual Camera Feeds</span>
                        <div class="view-switch hidden" id="viewSwitchGroup">
                            <button class="view-btn active" id="btnAnnotatedView" onclick="setViewMode('annotated')">Detections Overlay</button>
                            <button class="view-btn" id="btnOriginalView" onclick="setViewMode('original')">Original Photos</button>
                            <button class="view-btn" onclick="openFullscreenModal('dual')">⛶ Full Screen</button>
                            <button class="view-btn" onclick="downloadImage()">💾 Save Results</button>
                        </div>
                    </div>

                    <div class="display-area">
                        <!-- Dual Live Streams -->
                        <div class="dual-stream-container" id="dualStreamContainer">
                            <div class="stream-box">
                                <div class="stream-tag">Camera 1: Top View (Pose)</div>
                                <video id="webcam1" autoplay playsinline muted></video>
                            </div>
                            <div class="stream-box">
                                <div class="stream-tag">Camera 2: Side View (Origin 3)</div>
                                <video id="webcam2" autoplay playsinline muted></video>
                            </div>
                        </div>

                        <!-- Dual Upload Dropzones -->
                        <div class="dual-upload-container hidden" id="dualUploadContainer">
                            <div class="upload-half" onclick="document.getElementById('fileInputTop').click()">
                                <div class="stream-tag">Image 1: Top View (Wire Harness Pose)</div>
                                <div class="upload-dropzone" id="dropzoneTop">
                                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" style="margin-bottom:8px;">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                                    </svg>
                                    <h4 id="dropTopTitle" style="font-size:0.9rem; margin-bottom:4px;">Top View Image (Harness)</h4>
                                    <p id="dropTopSub" style="font-size:0.75rem; color:#94a3b8;">Click or Drag Top Wire Harness Photo</p>
                                    <img id="previewImgTop" class="upload-preview hidden" alt="Top View Preview">
                                </div>
                                <input type="file" id="fileInputTop" accept="image/*" class="hidden" onchange="handleFileSelectTop(event)">
                            </div>

                            <div class="upload-half" onclick="document.getElementById('fileInputOrig3').click()">
                                <div class="stream-tag">Image 2: Side View (Origin 3 Object)</div>
                                <div class="upload-dropzone" id="dropzoneOrig3">
                                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" style="margin-bottom:8px;">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                                    </svg>
                                    <h4 id="dropOrig3Title" style="font-size:0.9rem; margin-bottom:4px;">Side View Image (Origin 3)</h4>
                                    <p id="dropOrig3Sub" style="font-size:0.75rem; color:#94a3b8;">Click or Drag Origin 3 Side Photo</p>
                                    <img id="previewImgOrig3" class="upload-preview hidden" alt="Origin 3 Preview">
                                </div>
                                <input type="file" id="fileInputOrig3" accept="image/*" class="hidden" onchange="handleFileSelectOrig3(event)">
                            </div>
                        </div>

                        <!-- Dual Inspected Result Displays -->
                        <div class="dual-result-container hidden" id="dualResultContainer">
                            <div class="result-half" onclick="openFullscreenModal('top')" title="Click for Full Screen" style="cursor:zoom-in;">
                                <div class="stream-tag" id="tagResultTop">Top View: Pose Skeleton ⛶</div>
                                <img id="displayImgTop" class="result-display-img" alt="Top Pose Result">
                            </div>
                            <div class="result-half" onclick="openFullscreenModal('orig3')" title="Click for Full Screen" style="cursor:zoom-in;">
                                <div class="stream-tag" id="tagResultOrig3">Side View: Origin 3 Detection ⛶</div>
                                <img id="displayImgOrig3" class="result-display-img" alt="Origin 3 Result">
                            </div>
                        </div>

                        <!-- Canvas buffers for capturing frames -->
                        <canvas id="captureCanvas1" class="hidden"></canvas>
                        <canvas id="captureCanvas2" class="hidden"></canvas>

                        <!-- Shutter flash animation layer -->
                        <div class="shutter-flash" id="shutterFlash"></div>

                        <div class="camera-action-bar" id="cameraActionBar">
                            <button class="btn-start" id="btnSnap" onclick="startDualCapture()">
                                ▶ START INSPECTION
                            </button>
                            <button class="btn-retake hidden" id="btnRetake" onclick="resetToLiveCameras()">
                                🔄 Live Stream
                            </button>
                            <label style="display:flex; align-items:center; gap:8px; font-size:0.82rem; cursor:pointer; color:#94a3b8;">
                                <input type="checkbox" id="continuousToggle" onchange="toggleContinuous()">
                                Continuous Auto Mode
                            </label>
                        </div>

                        <div class="camera-action-bar hidden" id="uploadActionBar">
                            <button class="btn-snap" id="btnInspectUpload" onclick="inspectUploadedImages()">
                                🔍 Inspect Dual Images
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Results Panel -->
            <div class="results-panel">
                <div class="verdict-card ready" id="verdictCard">
                    <div class="verdict-title" id="verdictTitle">READY TO INSPECT</div>
                    <div class="verdict-subtitle" id="verdictSubtitle" style="font-size: 0.78rem; margin-top: 4px; color: #cbd5e1;">Both Camera 1 (Top Pose) & Camera 2 (Origin 3) must pass</div>
                    <div class="sub-verdicts-row" style="display:flex; gap:10px; justify-content:center; margin-top:8px;">
                        <span class="sub-verdict-pill" id="subVerdictPose">Top Pose: WAITING</span>
                        <span class="sub-verdict-pill" id="subVerdictOrig3">Origin 3: WAITING</span>
                    </div>
                    <div class="metrics-row" style="margin-top:8px;">
                        <span class="metric-item" id="latencyBadge">Total Latency: -- ms</span>
                        <span class="metric-item" id="wiresBadge">Wires: --</span>
                        <span class="metric-item" id="orig3ConfBadge">Origin 3 Conf: --</span>
                    </div>
                </div>

                <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">
                    System Component Checks
                </div>

                <!-- 0. Origin 3 Card (Side Camera) -->
                <div class="component-card" id="cardOrigin3">
                    <div class="comp-header">
                        <span class="comp-name">Origin 3 (Side Camera)</span>
                        <span class="comp-status-badge" id="badgeOrigin3">WAITING</span>
                    </div>
                    <div class="check-item">
                        <div class="check-top">
                            <span class="check-label">Component Presence</span>
                            <span class="check-pill" id="pillOrigin3">--</span>
                        </div>
                        <div class="check-rule">Rule: Origin 3 detected with confidence &ge; 40%</div>
                        <div class="check-detail" id="detailOrigin3">Waiting for side camera frame...</div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8; margin-top:4px; padding:0 4px;">
                        <span id="coordOrigin3">Center: --</span>
                        <span id="bboxOrigin3">BBox: --</span>
                    </div>
                </div>

                <!-- 1. UM Card -->
                <div class="component-card" id="cardUM">
                    <div class="comp-header">
                        <span class="comp-name">Upper Middle (Coils)</span>
                        <span class="comp-status-badge" id="badgeUM">WAITING</span>
                    </div>
                    <div class="check-item">
                        <div class="check-top">
                            <span class="check-label">Origin Position</span>
                            <span class="check-pill" id="pillUM_orig">--</span>
                        </div>
                        <div class="check-rule">Rule: Red origin left of Black (X_red &lt; X_black)</div>
                        <div class="check-detail" id="detailUM_orig">Waiting for frame...</div>
                    </div>
                    <div class="check-item">
                        <div class="check-top">
                            <span class="check-label">Terminal / Socket</span>
                            <span class="check-pill" id="pillUM_term">--</span>
                        </div>
                        <div class="check-rule">Rule: Red terminal right of Black (X_red &gt; X_black)</div>
                        <div class="check-detail" id="detailUM_term">Waiting for frame...</div>
                    </div>
                </div>

                <!-- 2. BL Card -->
                <div class="component-card" id="cardBL">
                    <div class="comp-header">
                        <span class="comp-name">Below Left (Terminals)</span>
                        <span class="comp-status-badge" id="badgeBL">WAITING</span>
                    </div>
                    <div class="check-item">
                        <div class="check-top">
                            <span class="check-label">Origin Position</span>
                            <span class="check-pill" id="pillBL_orig">--</span>
                        </div>
                        <div class="check-rule">Rule: Red origin below Black (Y_red &gt; Y_black)</div>
                        <div class="check-detail" id="detailBL_orig">Waiting for frame...</div>
                    </div>
                    <div class="check-item">
                        <div class="check-top">
                            <span class="check-label">Terminal / Socket</span>
                            <span class="check-pill" id="pillBL_term">--</span>
                        </div>
                        <div class="check-rule">Rule: Red terminal left of Black (X_red &lt; X_black)</div>
                        <div class="check-detail" id="detailBL_term">Waiting for frame...</div>
                    </div>
                </div>

                <!-- 3. BR Card -->
                <div class="component-card" id="cardBR">
                    <div class="comp-header">
                        <span class="comp-name">Below Right (Exit / Grommet)</span>
                        <span class="comp-status-badge" id="badgeBR">WAITING</span>
                    </div>
                    <div class="check-item">
                        <div class="check-top">
                            <span class="check-label">Origin Presence</span>
                            <span class="check-pill" id="pillBR_orig">--</span>
                        </div>
                        <div class="check-rule">Rule: Both wire origins exist</div>
                        <div class="check-detail" id="detailBR_orig">Waiting for frame...</div>
                    </div>
                    <div class="check-item">
                        <div class="check-top">
                            <span class="check-label">Terminal / Socket</span>
                            <span class="check-pill" id="pillBR_term">--</span>
                        </div>
                        <div class="check-rule">Rule: Red terminal left of Black (X_red &lt; X_black)</div>
                        <div class="check-detail" id="detailBR_term">Waiting for frame...</div>
                    </div>
                </div>

                <div class="defects-box hidden" id="defectsBox">
                    <h4>⚠️ Defects Detected</h4>
                    <ul id="defectsList"></ul>
                </div>
            </div>
        </div>
    </div>

    <!-- ================= FULLSCREEN LIGHTBOX MODAL ================= -->
    <div class="fs-modal hidden" id="fullscreenModal">
        <div class="fs-modal-backdrop" onclick="closeFullscreenModal()"></div>
        <div class="fs-modal-content">
            <div class="fs-modal-header">
                <div class="fs-header-left">
                    <span id="fsModalTitle" style="font-size: 1rem; font-weight: 800; color: #fff;">Dual Inspection Fullscreen</span>
                    <span class="sub-verdict-pill" id="fsVerdictPill">PASS</span>
                </div>
                <div class="fs-header-tabs">
                    <button class="view-btn active" id="fsTabDual" onclick="setFullscreenView('dual')">Dual Side-by-Side</button>
                    <button class="view-btn" id="fsTabTop" onclick="setFullscreenView('top')">Top View (Pose)</button>
                    <button class="view-btn" id="fsTabOrig3" onclick="setFullscreenView('orig3')">Side View (Origin 3)</button>
                    <button class="view-btn" id="fsTabToggleOverlay" onclick="toggleFullscreenOverlay()">Toggle Overlay</button>
                </div>
                <button class="fs-close-btn" onclick="closeFullscreenModal()" title="Close (ESC)">✕</button>
            </div>
            <div class="fs-image-body" id="fsImageBody">
                <div class="fs-dual-view" id="fsDualView">
                    <div class="fs-half">
                        <div class="stream-tag">Top View: Wire Harness Pose</div>
                        <img id="fsImgTop" src="" alt="Top View Fullscreen">
                    </div>
                    <div class="fs-half">
                        <div class="stream-tag">Side View: Origin 3 Detection</div>
                        <img id="fsImgOrig3" src="" alt="Origin 3 Fullscreen">
                    </div>
                </div>
                <div class="fs-single-view hidden" id="fsSingleView">
                    <img id="fsImgSingle" src="" alt="Single Fullscreen View">
                </div>
            </div>
            <div class="fs-modal-footer">
                <span id="fsFooterInfo" style="font-size:0.8rem; color:#94a3b8;">Press [ESC] to exit fullscreen | Click tabs to toggle views</span>
                <button class="view-btn" onclick="downloadImage()">💾 Download Images</button>
            </div>
        </div>
    </div>

    <!-- ================= VIEW 2: TOP MODEL (WIRE POSE) ANNOTATION & RETRAINING STUDIO ================= -->
    <div class="main-view" id="viewStudio">
        <div class="studio-workspace">
            
            <!-- Left Panel: Classes & Image Queue -->
            <div class="studio-card">
                <div class="card-header-title">
                    <span>Wire Classes</span>
                    <button class="btn-add-class" onclick="openAddClassModal()">+ Add Class</button>
                </div>

                <div class="class-chips-wrap" id="classChipsContainer">
                    <!-- Populated dynamically -->
                </div>

                <hr style="border-color: var(--border); margin: 6px 0;">

                <div class="card-header-title">
                    <span>Images to Annotate</span>
                    <button class="tool-btn" style="padding: 3px 8px; font-size: 0.72rem;" onclick="document.getElementById('studioUploadInput').click()">
                        + Upload
                    </button>
                    <input type="file" id="studioUploadInput" accept="image/*" multiple class="hidden" onchange="handleStudioUpload(event)">
                </div>

                <div style="display: flex; gap: 6px; font-size: 0.74rem;">
                    <button class="tool-btn" style="flex:1; padding:4px;" onclick="snapCameraToAnnotate()">📸 From Cam</button>
                    <button class="tool-btn" style="flex:1; padding:4px;" onclick="fetchStudioImages()">🔄 Refresh</button>
                </div>

                <div class="image-list-box" id="studioImageList">
                    <!-- Loaded dynamically -->
                </div>
            </div>

            <!-- Center Panel: Interactive Keypoint Canvas -->
            <div class="studio-card" id="annoCenterCard" style="padding: 10px;">
                <div class="canvas-toolbar">
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span id="canvasStepNotice" style="font-weight: 700; color: #60a5fa; font-size: 0.85rem;">
                            Step 1: Click Wire Origin (Coil Connection)
                        </span>
                        <span id="currentImgBadge" style="font-size: 0.72rem; color: #cbd5e1; background: var(--bg-card); padding: 2px 8px; border-radius: 4px; border: 1px solid var(--border);">No image loaded</span>
                    </div>

                    <!-- Class Selector Toolbar (Dropdown + Tick boxes) -->
                    <div class="anno-class-selector-bar">
                        <span style="font-size: 0.74rem; font-weight: 700; color: #94a3b8; display: flex; align-items: center; gap: 4px;">
                            <span>Wire:</span>
                        </span>
                        <!-- Dropdown Selector -->
                        <select id="annoClassSelect" class="anno-class-select" onchange="setActiveClass(this.value)" title="Select active wire class via dropdown">
                            <!-- Populated dynamically -->
                        </select>
                        <span style="color: #475569; font-size: 0.8rem;">|</span>
                        <!-- Tick Box / Radio Selection -->
                        <div id="annoClassTickGroup" class="anno-class-tick-group">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                        <button class="tool-btn" id="btnToggleFs" onclick="toggleFullscreenAnnotation()">⛶ Fullscreen</button>
                        <button class="tool-btn" onclick="undoPoint()">↩️ Undo Point</button>
                        <button class="tool-btn" onclick="clearCurrentWire()">🗑️ Clear Wire</button>
                        <button class="tool-btn" style="color: #f87171;" onclick="deleteCurrentImage()">🗑️ Delete Image</button>
                        <button class="tool-btn primary" onclick="saveAnnotation()">💾 Save Annotation</button>
                    </div>
                </div>

                <div class="canvas-container" id="canvasContainer">
                    <canvas id="annoCanvas"></canvas>
                </div>

                <div id="finishedWiresSection" class="finished-wires-box">
                    <div class="finished-wires-header">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 0.74rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">
                                📋 Labeled Wires:
                            </span>
                            <span id="finishedWiresBadge" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); padding: 1px 7px; border-radius: 10px; font-size: 0.7rem; font-weight: 700;">
                                0
                            </span>
                        </div>
                        <button id="btnToggleWiresList" class="btn-toggle-wires" onclick="toggleWiresCollapse()" title="Collapse / Expand labeled wires list">
                            ▼ Hide Wires
                        </button>
                    </div>
                    <div id="finishedWiresList" class="finished-wires-body">
                        <span style="color:#64748b; font-size:0.75rem;">No wires labeled yet. Click 3 points per wire.</span>
                    </div>
                </div>
            </div>

            <!-- Right Panel: Retraining Management -->
            <div class="studio-card">
                <div class="card-header-title">
                    <span>Model Retraining</span>
                    <span class="badge badge-model" id="trainStatusBadge">Idle</span>
                </div>

                <div style="background: var(--bg-card); border-radius: 6px; padding: 10px; font-size: 0.78rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: var(--text-muted);">Dataset Images:</span>
                        <b id="dsTotalImages">--</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: var(--text-muted);">Active Classes:</span>
                        <b id="dsTotalClasses">--</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-muted);">Device:</span>
                        <b>CPU (Optimized)</b>
                    </div>
                </div>

                <div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.78rem;">
                    <label>Training Epochs:</label>
                    <input type="number" id="trainEpochsInput" min="5" max="100" value="15">

                    <label>Batch Size:</label>
                    <input type="number" id="trainBatchInput" min="2" max="32" value="8">

                    <label>Base Weights:</label>
                    <select id="trainWeightsSelect">
                        <option value="best">Continue from Current best.pt (Fine-tune)</option>
                        <option value="nano">Train Fresh from yolov8n-pose.pt</option>
                    </select>
                </div>

                <button class="btn-snap" id="btnStartTrain" style="justify-content: center; width: 100%; border-radius: 6px;" onclick="startRetraining()">
                    🚀 Retrain Model
                </button>

                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted);">
                        <span>Training Progress</span>
                        <span id="trainProgressText">0%</span>
                    </div>
                    <div class="progress-bar-wrap">
                        <div class="progress-bar-fill" id="trainProgressBar"></div>
                    </div>
                </div>

                <div class="retrain-console" id="trainConsole">Training console waiting for run...</div>
            </div>

        </div>
    </div>

    <!-- ================= VIEW 3: SIDE MODEL (ORIGIN 3) ANNOTATION & RETRAINING STUDIO ================= -->
    <div class="main-view" id="viewSideStudio">
        <div class="studio-workspace">
            
            <!-- Left Panel: Origin 3 Target & Side Images Queue -->
            <div class="studio-card">
                <div class="card-header-title">
                    <span>Origin 3 Target Class</span>
                    <span class="badge" style="background: rgba(6, 182, 212, 0.18); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.4);">
                        Class 0: origin_3
                    </span>
                </div>

                <div style="background: var(--bg-card); border-radius: 6px; padding: 8px 10px; font-size: 0.76rem; border: 1px solid var(--border);">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="width: 10px; height: 10px; border-radius: 50%; background: #06b6d4; display: inline-block;"></span>
                        <b style="color: #f8fafc;">origin_3</b>
                        <span style="color: var(--text-muted); font-size: 0.7rem;">(YOLO Object Detection)</span>
                    </div>
                    <p style="color: var(--text-muted); font-size: 0.72rem; line-height: 1.35;">
                        Click and drag a bounding box enclosing the Origin 3 connector terminal.
                    </p>
                </div>

                <hr style="border-color: var(--border); margin: 4px 0;">

                <div class="card-header-title">
                    <span>Side View Images</span>
                    <button class="tool-btn" style="padding: 3px 8px; font-size: 0.72rem;" onclick="document.getElementById('sideUploadInput').click()">
                        + Upload
                    </button>
                    <input type="file" id="sideUploadInput" accept="image/*" multiple class="hidden" onchange="handleSideUpload(event)">
                </div>

                <div style="display: flex; gap: 6px; font-size: 0.74rem;">
                    <button class="tool-btn" style="flex:1; padding:4px;" onclick="snapSideCamera()">📸 Cam 2 (Side)</button>
                    <button class="tool-btn" style="flex:1; padding:4px;" onclick="fetchSideImages()">🔄 Refresh</button>
                </div>

                <!-- Filter Chips -->
                <div class="filter-tabs-row">
                    <button class="filter-pill active" id="sideFilterBtnAll" onclick="setSideFilter('all')">All (<span id="sideCountAll">0</span>)</button>
                    <button class="filter-pill" id="sideFilterBtnLabeled" onclick="setSideFilter('labeled')">Labeled (<span id="sideCountLabeled">0</span>)</button>
                    <button class="filter-pill" id="sideFilterBtnUnlabeled" onclick="setSideFilter('unlabeled')">Unlabeled (<span id="sideCountUnlabeled">0</span>)</button>
                </div>

                <div class="image-list-box" id="sideImageList" style="max-height: 440px;">
                    <!-- Loaded dynamically -->
                </div>
            </div>

            <!-- Center Panel: Interactive Bounding Box Canvas -->
            <div class="studio-card" id="sideCenterCard" style="padding: 10px;">
                <div class="canvas-toolbar">
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span id="sideCanvasStatus" style="font-weight: 700; color: #38bdf8; font-size: 0.85rem;">
                            Target: Origin 3 &bull; Drag to Draw Box
                        </span>
                        <span id="sideCurrentImgBadge" style="font-size: 0.72rem; color: #cbd5e1; background: var(--bg-card); padding: 2px 8px; border-radius: 4px; border: 1px solid var(--border);">
                            No image loaded
                        </span>
                    </div>

                    <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                        <button class="tool-btn" style="color: #38bdf8; border-color: rgba(6, 182, 212, 0.4);" onclick="sidePreAnnotate()" title="Run current Origin 3 model to auto-detect box">
                            🤖 Auto-Detect
                        </button>
                        <button class="tool-btn" id="btnToggleSideFs" onclick="toggleFullscreenSideAnnotation()">⛶ Fullscreen</button>
                        <button class="tool-btn" onclick="clearSideBox()">↩️ Clear Box</button>
                        <button class="tool-btn" style="color: #f87171;" onclick="deleteCurrentSideImage()">🗑️ Delete Image</button>
                        <button class="tool-btn primary" style="background: #0284c7; border-color: #0284c7;" onclick="saveSideAnnotation()">💾 Save Annotation</button>
                    </div>
                </div>

                <div class="canvas-container" id="sideCanvasContainer">
                    <canvas id="sideCanvas"></canvas>
                </div>

                <!-- Bounding Box Details -->
                <div class="finished-wires-box" id="sideBoxDetailsBox">
                    <div class="finished-wires-header">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 0.74rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">
                                📦 Origin 3 Bounding Box:
                            </span>
                            <span id="sideBoxBadge" style="background: rgba(6, 182, 212, 0.2); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.4); padding: 1px 7px; border-radius: 10px; font-size: 0.7rem; font-weight: 700;">
                                None
                            </span>
                        </div>
                        <span id="sideBoxCoords" style="color: #64748b; font-size: 0.75rem; font-family: monospace;">
                            No box drawn yet. Drag rectangle across Origin 3.
                        </span>
                    </div>
                </div>
            </div>

            <!-- Right Panel: Origin 3 Retraining Management -->
            <div class="studio-card">
                <div class="card-header-title">
                    <span>Origin 3 Retraining</span>
                    <span class="badge badge-model" id="sideTrainStatusBadge">Idle</span>
                </div>

                <div style="background: var(--bg-card); border-radius: 6px; padding: 10px; font-size: 0.78rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: var(--text-muted);">Dataset Images:</span>
                        <b id="sideDsTotal">--</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: var(--text-muted);">Labeled / Unlabeled:</span>
                        <span><b id="sideDsLabeled" style="color: #34d399;">--</b> / <b id="sideDsUnlabeled" style="color: #f59e0b;">--</b></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: var(--text-muted);">Split:</span>
                        <b>80% Train / 20% Val</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-muted);">Device:</span>
                        <b id="sideDsDevice">CPU (Optimized)</b>
                    </div>
                </div>

                <div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.78rem;">
                    <label>Training Epochs:</label>
                    <input type="number" id="sideTrainEpochsInput" min="5" max="150" value="30">

                    <label>Batch Size:</label>
                    <input type="number" id="sideTrainBatchInput" min="2" max="32" value="8">

                    <label>Base Weights:</label>
                    <select id="sideTrainWeightsSelect" class="anno-class-select" style="width: 100%;">
                        <option value="best">Continue from Current best.pt (Fine-tune)</option>
                        <option value="nano">Train Fresh from yolov8n.pt</option>
                    </select>
                </div>

                <button class="btn-snap" id="btnStartSideTrain" style="justify-content: center; width: 100%; border-radius: 6px; background: #0284c7;" onclick="startSideRetraining()">
                    🚀 Retrain Side Model (Origin 3)
                </button>

                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted);">
                        <span>Training Progress</span>
                        <span id="sideTrainProgressText">0%</span>
                    </div>
                    <div class="progress-bar-wrap">
                        <div class="progress-bar-fill" id="sideTrainProgressBar" style="background: linear-gradient(90deg, #0284c7, #06b6d4, #10b981);"></div>
                    </div>
                </div>

                <div class="retrain-console" id="sideTrainConsole" style="color: #67e8f9;">Origin 3 retraining console waiting for run...</div>
            </div>

        </div>
    </div>

    <!-- Modal: Add New Class -->
    <div class="modal-overlay hidden" id="addClassModal">
        <div class="modal-box">
            <h3 style="font-size: 1rem;">Add New Wire Class</h3>
            <p style="font-size: 0.75rem; color: var(--text-muted);">Define a new wire color or type for the model:</p>
            
            <label style="font-size: 0.78rem;">Class Name:</label>
            <input type="text" id="newClassNameInput" placeholder="e.g. wire_blue or wire_yellow">

            <label style="font-size: 0.78rem;">Display Color:</label>
            <input type="color" id="newClassColorInput" value="#3b82f6" style="width: 100%; height: 38px; border: none; background: transparent; cursor: pointer;">

            <div style="display: flex; justify-content: flex-end; gap: 8px; margin-top: 6px;">
                <button class="tool-btn" onclick="closeAddClassModal()">Cancel</button>
                <button class="tool-btn primary" onclick="submitAddClass()">Save Class</button>
            </div>
        </div>
    </div>

    <canvas id="captureCanvas" class="hidden"></canvas>

    <script>
        // State
        let currentTab = "inspect";
        let activeStream = null;
        let continuousTimer = null;
        let isInspecting = false;
        let currentResult = null;
        let viewMode = "annotated";

        // Annotation State
        let studioClasses = [];
        let activeClassId = 0;
        let currentImageName = "";
        let currentImageLoaded = null;
        let annoWires = []; // list of { class_id: int, points: [[x,y],[x,y],[x,y]] }
        let pendingPoints = []; // current points being placed: [(x,y)]
        let imgOrigW = 1280, imgOrigH = 720;

        // Elements
        let activeStream1 = null;
        let activeStream2 = null;
        let uploadedTopB64 = null;
        let uploadedOrig3B64 = null;

        // Elements
        const webcam1 = document.getElementById('webcam1');
        const webcam2 = document.getElementById('webcam2');
        const dualStreamContainer = document.getElementById('dualStreamContainer');
        const dualUploadContainer = document.getElementById('dualUploadContainer');
        const dualResultContainer = document.getElementById('dualResultContainer');
        const displayImgTop = document.getElementById('displayImgTop');
        const displayImgOrig3 = document.getElementById('displayImgOrig3');
        const cameraActionBar = document.getElementById('cameraActionBar');
        const uploadActionBar = document.getElementById('uploadActionBar');
        const cameraSelectWrap = document.getElementById('cameraSelectWrap');
        const cameraSelect1 = document.getElementById('cameraSelect1');
        const cameraSelect2 = document.getElementById('cameraSelect2');
        const viewSwitchGroup = document.getElementById('viewSwitchGroup');
        const confSlider = document.getElementById('confSlider');
        const confVal = document.getElementById('confVal');
        const annoCanvas = document.getElementById('annoCanvas');
        const ctxAnno = annoCanvas.getContext('2d');

        confSlider.addEventListener('input', (e) => {
            confVal.textContent = (e.target.value / 100).toFixed(2);
        });

        function switchMainTab(tab) {
            if (tab === 'studio') tab = 'top_studio';
            currentTab = tab;

            const btnInspect = document.getElementById('btnNavInspect');
            const btnTop = document.getElementById('btnNavTopStudio');
            const btnSide = document.getElementById('btnNavSideStudio');

            if (btnInspect) btnInspect.classList.toggle('active', tab === 'inspect');
            if (btnTop) btnTop.classList.toggle('active', tab === 'top_studio');
            if (btnSide) btnSide.classList.toggle('active', tab === 'side_studio');

            const viewInspect = document.getElementById('viewInspect');
            const viewStudio = document.getElementById('viewStudio');
            const viewSideStudio = document.getElementById('viewSideStudio');

            if (viewInspect) viewInspect.classList.toggle('active', tab === 'inspect');
            if (viewStudio) viewStudio.classList.toggle('active', tab === 'top_studio');
            if (viewSideStudio) viewSideStudio.classList.toggle('active', tab === 'side_studio');

            if (tab === 'top_studio') {
                loadStudioData();
            } else if (tab === 'side_studio') {
                loadSideStudioData();
            } else {
                if (document.getElementById('tabCamera') && document.getElementById('tabCamera').classList.contains('active')) {
                    if (activeStream1) webcam1.play();
                    if (activeStream2) webcam2.play();
                }
            }
        }

        // ================= DUAL INSPECTION METHODS =================
        async function initCameras() {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(d => d.kind === 'videoinput');
                cameraSelect1.innerHTML = '';
                cameraSelect2.innerHTML = '';

                if (videoDevices.length === 0) {
                    cameraSelect1.innerHTML = '<option value="">No webcam detected</option>';
                    cameraSelect2.innerHTML = '<option value="">No webcam detected</option>';
                    return;
                }

                videoDevices.forEach((dev, idx) => {
                    const opt1 = document.createElement('option');
                    opt1.value = dev.deviceId;
                    opt1.text = dev.label || `Camera ${idx + 1}`;
                    cameraSelect1.appendChild(opt1);

                    const opt2 = document.createElement('option');
                    opt2.value = dev.deviceId;
                    opt2.text = dev.label || `Camera ${idx + 1}`;
                    cameraSelect2.appendChild(opt2);
                });

                // Default camera 2 to second camera if available
                if (videoDevices.length > 1) {
                    cameraSelect2.selectedIndex = 1;
                }

                startCameras();
            } catch (err) {
                console.error("Camera access error:", err);
            }
        }

        async function startCameras() {
            stopStreams();
            const devId1 = cameraSelect1.value;
            const devId2 = cameraSelect2.value;

            try {
                if (devId1) {
                    const stream1 = await navigator.mediaDevices.getUserMedia({
                        video: { deviceId: { exact: devId1 }, width: { ideal: 1280 }, height: { ideal: 720 } }
                    });
                    activeStream1 = stream1;
                    webcam1.srcObject = stream1;
                }
            } catch (e) {
                console.warn("Could not start Cam 1:", e);
            }

            try {
                if (devId2 && devId2 !== devId1) {
                    const stream2 = await navigator.mediaDevices.getUserMedia({
                        video: { deviceId: { exact: devId2 }, width: { ideal: 1280 }, height: { ideal: 720 } }
                    });
                    activeStream2 = stream2;
                    webcam2.srcObject = stream2;
                } else if (devId2) {
                    if (activeStream1) {
                        webcam2.srcObject = activeStream1;
                    }
                }
            } catch (e) {
                console.warn("Could not start Cam 2:", e);
            }

            dualStreamContainer.classList.remove('hidden');
            dualResultContainer.classList.add('hidden');
            dualUploadContainer.classList.add('hidden');
        }

        function stopStreams() {
            if (activeStream1) {
                activeStream1.getTracks().forEach(t => t.stop());
                activeStream1 = null;
            }
            if (activeStream2) {
                activeStream2.getTracks().forEach(t => t.stop());
                activeStream2 = null;
            }
        }

        function switchInspectMode(mode) {
            stopContinuous();
            if (mode === 'camera') {
                document.getElementById('tabCamera').classList.add('active');
                document.getElementById('tabUpload').classList.remove('active');
                cameraSelectWrap.classList.remove('hidden');
                cameraActionBar.classList.remove('hidden');
                uploadActionBar.classList.add('hidden');
                document.getElementById('samplesBar').classList.add('hidden');
                dualUploadContainer.classList.add('hidden');
                dualResultContainer.classList.add('hidden');
                dualStreamContainer.classList.remove('hidden');
                document.getElementById('stageTitle').textContent = "Live Dual Camera Feeds";
                viewSwitchGroup.classList.add('hidden');
                startCameras();
            } else {
                document.getElementById('tabUpload').classList.add('active');
                document.getElementById('tabCamera').classList.remove('active');
                cameraSelectWrap.classList.add('hidden');
                cameraActionBar.classList.add('hidden');
                uploadActionBar.classList.remove('hidden');
                document.getElementById('samplesBar').classList.remove('hidden');
                stopStreams();
                dualStreamContainer.classList.add('hidden');
                if (!currentResult) {
                    dualUploadContainer.classList.remove('hidden');
                    dualResultContainer.classList.add('hidden');
                    document.getElementById('stageTitle').textContent = "Upload Top & Side Images";
                } else {
                    dualUploadContainer.classList.add('hidden');
                    dualResultContainer.classList.remove('hidden');
                }
            }
        }

        function triggerShutterFlash() {
            const flash = document.getElementById('shutterFlash');
            if (flash) {
                flash.classList.add('flash');
                setTimeout(() => flash.classList.remove('flash'), 180);
            }
        }

        function resetToLiveCameras() {
            dualStreamContainer.classList.remove('hidden');
            dualResultContainer.classList.add('hidden');
            const btnRetake = document.getElementById('btnRetake');
            if (btnRetake) btnRetake.classList.add('hidden');
            document.getElementById('stageTitle').textContent = "Live Dual Camera Feeds";
            if (activeStream1) webcam1.play();
            if (activeStream2) webcam2.play();
        }

        async function startDualCapture() {
            if (isInspecting) return;

            // If we are showing results, restore live view first
            if (!dualResultContainer.classList.contains('hidden')) {
                resetToLiveCameras();
                await new Promise(r => setTimeout(r, 120));
            }

            // Trigger visual camera shutter flash
            triggerShutterFlash();

            const c1 = document.getElementById('captureCanvas1');
            const c2 = document.getElementById('captureCanvas2');
            
            c1.width = webcam1.videoWidth || 1280;
            c1.height = webcam1.videoHeight || 720;
            const ctx1 = c1.getContext('2d');
            ctx1.drawImage(webcam1, 0, 0, c1.width, c1.height);
            const b64_1 = c1.toDataURL('image/jpeg', 0.92);

            c2.width = webcam2.videoWidth || 1280;
            c2.height = webcam2.videoHeight || 720;
            const ctx2 = c2.getContext('2d');
            ctx2.drawImage(webcam2, 0, 0, c2.width, c2.height);
            const b64_2 = c2.toDataURL('image/jpeg', 0.92);

            const btn = document.getElementById('btnSnap');
            if (btn) {
                btn.innerHTML = '⏳ Capturing & Inspecting...';
                btn.disabled = true;
            }

            try {
                await runInspection({ top_image_base64: b64_1, orig3_image_base64: b64_2 });
                const btnRetake = document.getElementById('btnRetake');
                if (btnRetake) btnRetake.classList.remove('hidden');
            } finally {
                if (btn) {
                    btn.innerHTML = '▶ START INSPECTION';
                    btn.disabled = false;
                }
            }
        }

        async function snapAndInspect() {
            await startDualCapture();
        }

        function toggleContinuous() {
            const chk = document.getElementById('continuousToggle');
            if (chk.checked) {
                continuousTimer = setInterval(() => {
                    if (!isInspecting && document.getElementById('tabCamera').classList.contains('active')) {
                        snapAndInspect();
                    }
                }, 1500);
            } else {
                stopContinuous();
            }
        }

        function stopContinuous() {
            if (continuousTimer) {
                clearInterval(continuousTimer);
                continuousTimer = null;
            }
            document.getElementById('continuousToggle').checked = false;
        }

        async function loadSamplePair(type) {
            stopContinuous();
            if (type === 'pass') {
                await runInspection({
                    sample_name_top: 'correctwiring (1).jpg',
                    sample_name_orig3: 'orig3_pass.jpg'
                });
            } else if (type === 'fail_pose') {
                await runInspection({
                    sample_name_top: 'fail_case1_bl.jpg',
                    sample_name_orig3: 'orig3_pass.jpg'
                });
            } else if (type === 'fail_orig3') {
                await runInspection({
                    sample_name_top: 'correctwiring (1).jpg',
                    sample_name_orig3: 'orig3_missing.jpg'
                });
            }
        }

        function handleFileSelectTop(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (ev) => {
                    uploadedTopB64 = ev.target.result;
                    const prev = document.getElementById('previewImgTop');
                    prev.src = uploadedTopB64;
                    prev.classList.remove('hidden');
                    document.getElementById('dropTopTitle').textContent = file.name;
                    document.getElementById('dropTopSub').textContent = "Loaded (Click to change)";
                };
                reader.readAsDataURL(file);
            }
        }

        function handleFileSelectOrig3(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (ev) => {
                    uploadedOrig3B64 = ev.target.result;
                    const prev = document.getElementById('previewImgOrig3');
                    prev.src = uploadedOrig3B64;
                    prev.classList.remove('hidden');
                    document.getElementById('dropOrig3Title').textContent = file.name;
                    document.getElementById('dropOrig3Sub').textContent = "Loaded (Click to change)";
                };
                reader.readAsDataURL(file);
            }
        }

        async function inspectUploadedImages() {
            if (!uploadedTopB64 && !uploadedOrig3B64) {
                alert("Please select or drop at least one image (Top View or Origin 3 Side View).");
                return;
            }
            await runInspection({
                top_image_base64: uploadedTopB64,
                orig3_image_base64: uploadedOrig3B64
            });
        }

        async function runInspection(payload) {
            isInspecting = true;
            payload.conf = parseFloat(confSlider.value) / 100.0;
            payload.conf_orig3 = 0.40;
            const btnSnap = document.getElementById('btnSnap');
            const btnUpload = document.getElementById('btnInspectUpload');
            if (btnSnap) btnSnap.innerHTML = 'Inspecting Both Cameras...';
            if (btnUpload) btnUpload.innerHTML = 'Inspecting Dual Images...';

            try {
                const resp = await fetch('/api/inspect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();
                if (data.success) {
                    renderResults(data);
                } else {
                    alert("Inspection error: " + (data.error || "Failed"));
                }
            } catch (err) {
                console.error("API Error:", err);
            } finally {
                isInspecting = false;
                if (btnSnap) btnSnap.innerHTML = '📸 Snap & Inspect Both Cameras';
                if (btnUpload) btnUpload.innerHTML = '🔍 Inspect Dual Images';
            }
        }

        function renderResults(data) {
            currentResult = data;
            const isPass = data.overall_pass;
            const vCard = document.getElementById('verdictCard');
            const vTitle = document.getElementById('verdictTitle');
            const vSub = document.getElementById('verdictSubtitle');

            // 1. Overall Verdict Banner
            vCard.className = "verdict-card " + (isPass ? "pass" : "fail");
            vTitle.innerHTML = isPass ? "✔ OVERALL: PASS" : "✘ OVERALL: FAIL";
            vSub.textContent = isPass 
                ? "Both Top Wire Harness & Origin 3 verified successfully!"
                : (data.reasons_failed.length + " check(s) failed across system.");

            // Sub-verdict pills
            const subPose = document.getElementById('subVerdictPose');
            const subOrig3 = document.getElementById('subVerdictOrig3');
            if (subPose) {
                subPose.className = "sub-verdict-pill " + (data.pose_pass ? "pass" : "fail");
                subPose.textContent = "Top Pose: " + (data.pose_pass ? "PASS" : "FAIL");
            }
            if (subOrig3) {
                subOrig3.className = "sub-verdict-pill " + (data.orig3_pass ? "pass" : "fail");
                subOrig3.textContent = "Origin 3: " + (data.orig3_pass ? "PASS" : "FAIL");
            }

            document.getElementById('latencyBadge').textContent = `Total Latency: ${data.latency_ms} ms`;
            document.getElementById('wiresBadge').textContent = `Wires: ${data.total_wires} detected`;
            const oConf = data.origin3_card ? data.origin3_card.confidence : 0;
            document.getElementById('orig3ConfBadge').textContent = `Origin 3 Conf: ${oConf}%`;

            // Display containers
            viewSwitchGroup.classList.remove('hidden');
            dualStreamContainer.classList.add('hidden');
            dualUploadContainer.classList.add('hidden');
            dualResultContainer.classList.remove('hidden');
            updateDisplayImage();

            // 2. Origin 3 Component Card
            const o3 = data.origin3_card || {};
            const cardO3 = document.getElementById('cardOrigin3');
            const badgeO3 = document.getElementById('badgeOrigin3');
            const pillO3 = document.getElementById('pillOrigin3');
            const detailO3 = document.getElementById('detailOrigin3');
            const coordO3 = document.getElementById('coordOrigin3');
            const bboxO3 = document.getElementById('bboxOrigin3');

            if (cardO3) cardO3.className = "component-card " + (o3.is_pass ? "pass" : "fail");
            if (badgeO3) {
                badgeO3.className = "comp-status-badge " + (o3.is_pass ? "pass" : "fail");
                badgeO3.textContent = o3.is_pass ? "PASS" : "FAIL";
            }
            if (pillO3) {
                pillO3.className = "check-pill " + (o3.is_pass ? "pass" : "fail");
                pillO3.textContent = o3.is_pass ? "✔ PASS" : "✘ FAIL";
            }
            if (detailO3) detailO3.textContent = o3.msg || "Not evaluated";
            if (coordO3) coordO3.textContent = o3.center ? `Center: (${o3.center[0]}, ${o3.center[1]})` : "Center: --";
            if (bboxO3) bboxO3.textContent = o3.bbox ? `BBox: [${o3.bbox.join(', ')}]` : "BBox: --";

            // 3. Pose Component Cards (UM, BL, BR)
            data.components.forEach(comp => {
                const card = document.getElementById('card' + comp.id);
                const badge = document.getElementById('badge' + comp.id);
                if (!card) return;

                card.className = "component-card " + (comp.pass ? "pass" : "fail");
                badge.className = "comp-status-badge " + (comp.pass ? "pass" : "fail");
                badge.textContent = comp.pass ? "PASS" : "FAIL";

                const origCheck = comp.checks[0];
                const pOrig = document.getElementById('pill' + comp.id + '_orig');
                const dOrig = document.getElementById('detail' + comp.id + '_orig');
                if (pOrig) {
                    pOrig.className = "check-pill " + (origCheck.pass ? "pass" : "fail");
                    pOrig.textContent = origCheck.pass ? "✔ PASS" : "✘ FAIL";
                }
                if (dOrig) dOrig.textContent = origCheck.detail;

                const termCheck = comp.checks[1];
                const pTerm = document.getElementById('pill' + comp.id + '_term');
                const dTerm = document.getElementById('detail' + comp.id + '_term');
                if (pTerm) {
                    pTerm.className = "check-pill " + (termCheck.pass ? "pass" : "fail");
                    pTerm.textContent = termCheck.pass ? "✔ PASS" : "✘ FAIL";
                }
                if (dTerm) dTerm.textContent = termCheck.detail;
            });

            // 4. Combined Defects Box
            const defBox = document.getElementById('defectsBox');
            const defList = document.getElementById('defectsList');
            if (!isPass && data.reasons_failed && data.reasons_failed.length > 0) {
                defBox.classList.remove('hidden');
                defList.innerHTML = data.reasons_failed.map(r => `<li>${r}</li>`).join('');
            } else {
                defBox.classList.add('hidden');
            }
        }

        function setViewMode(mode) {
            viewMode = mode;
            document.getElementById('btnAnnotatedView').classList.toggle('active', mode === 'annotated');
            document.getElementById('btnOriginalView').classList.toggle('active', mode === 'original');
            updateDisplayImage();
        }

        function updateDisplayImage() {
            if (!currentResult) return;
            if (viewMode === 'annotated') {
                displayImgTop.src = currentResult.annotated_image_top || currentResult.annotated_image;
                displayImgOrig3.src = currentResult.annotated_image_orig3 || "";
            } else {
                displayImgTop.src = currentResult.original_image_top || currentResult.original_image;
                displayImgOrig3.src = currentResult.original_image_orig3 || "";
            }
        }

        function downloadImage() {
            if (!currentResult) return;
            const a1 = document.createElement('a');
            a1.href = currentResult.annotated_image_top;
            a1.download = `top_inspected_${Date.now()}.jpg`;
            a1.click();

            if (currentResult.annotated_image_orig3) {
                setTimeout(() => {
                    const a2 = document.createElement('a');
                    a2.href = currentResult.annotated_image_orig3;
                    a2.download = `orig3_inspected_${Date.now()}.jpg`;
                    a2.click();
                }, 300);
            }
        }

        // ================= FULLSCREEN LIGHTBOX METHODS =================
        let fsCurrentView = 'dual'; // 'dual', 'top', 'orig3'
        let fsOverlayMode = 'annotated'; // 'annotated', 'original'

        function openFullscreenModal(initialView) {
            if (!currentResult) {
                alert("No inspection result available yet. Please click START or load a sample first.");
                return;
            }
            fsCurrentView = initialView || 'dual';
            fsOverlayMode = viewMode || 'annotated';

            const modal = document.getElementById('fullscreenModal');
            modal.classList.remove('hidden');

            const pill = document.getElementById('fsVerdictPill');
            const isPass = currentResult.overall_pass;
            pill.className = "sub-verdict-pill " + (isPass ? "pass" : "fail");
            pill.textContent = isPass ? "✔ PASS" : "✘ FAIL";

            setFullscreenView(fsCurrentView);
        }

        function closeFullscreenModal() {
            const modal = document.getElementById('fullscreenModal');
            if (modal) modal.classList.add('hidden');
        }

        function setFullscreenView(view) {
            fsCurrentView = view;
            document.getElementById('fsTabDual').classList.toggle('active', view === 'dual');
            document.getElementById('fsTabTop').classList.toggle('active', view === 'top');
            document.getElementById('fsTabOrig3').classList.toggle('active', view === 'orig3');

            const dualBox = document.getElementById('fsDualView');
            const singleBox = document.getElementById('fsSingleView');
            const title = document.getElementById('fsModalTitle');
            const imgSingle = document.getElementById('fsImgSingle');
            const imgTop = document.getElementById('fsImgTop');
            const imgOrig3 = document.getElementById('fsImgOrig3');

            const topSrc = fsOverlayMode === 'annotated' ? (currentResult.annotated_image_top || currentResult.annotated_image) : (currentResult.original_image_top || currentResult.original_image);
            const orig3Src = fsOverlayMode === 'annotated' ? currentResult.annotated_image_orig3 : currentResult.original_image_orig3;

            if (view === 'dual') {
                dualBox.classList.remove('hidden');
                singleBox.classList.add('hidden');
                imgTop.src = topSrc || "";
                imgOrig3.src = orig3Src || "";
                title.textContent = "Dual Inspection Fullscreen (Top + Side View)";
            } else if (view === 'top') {
                dualBox.classList.add('hidden');
                singleBox.classList.remove('hidden');
                imgSingle.src = topSrc || "";
                title.textContent = "Top View (Wire Harness Pose) - Fullscreen";
            } else if (view === 'orig3') {
                dualBox.classList.add('hidden');
                singleBox.classList.remove('hidden');
                imgSingle.src = orig3Src || "";
                title.textContent = "Side View (Origin 3 Detection) - Fullscreen";
            }
        }

        function toggleFullscreenOverlay() {
            fsOverlayMode = fsOverlayMode === 'annotated' ? 'original' : 'annotated';
            const btn = document.getElementById('fsTabToggleOverlay');
            btn.textContent = fsOverlayMode === 'annotated' ? "Toggle Overlay (Showing: Overlay)" : "Toggle Overlay (Showing: Original)";
            setFullscreenView(fsCurrentView);
        }

        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeFullscreenModal();
            } else if (e.key === 'f' || e.key === 'F') {
                if (['INPUT', 'TEXTAREA'].indexOf(document.activeElement.tagName) === -1) {
                    const modal = document.getElementById('fullscreenModal');
                    if (modal && !modal.classList.contains('hidden')) {
                        closeFullscreenModal();
                    } else if (currentResult) {
                        openFullscreenModal('dual');
                    }
                }
            }
        });

        // ================= 2. ANNOTATION STUDIO METHODS =================
        async function loadStudioData() {
            await fetchClasses();
            await fetchStudioImages();
            pollTrainingStatus();
        }

        async function fetchClasses() {
            try {
                const res = await fetch('/api/classes');
                studioClasses = await res.json();
                renderClassChips();
                document.getElementById('dsTotalClasses').textContent = studioClasses.length;
            } catch (err) {
                console.error("Classes load error:", err);
            }
        }

        function setActiveClass(clsId) {
            activeClassId = parseInt(clsId);
            updateClassSelectors();
            updateStepNotice();
            redrawCanvas();
        }

        function updateClassSelectors() {
            // 1. Update left panel chips
            const chipContainer = document.getElementById('classChipsContainer');
            if (chipContainer) {
                chipContainer.querySelectorAll('.class-chip').forEach(chip => {
                    const cid = parseInt(chip.getAttribute('data-class-id'));
                    chip.classList.toggle('active', cid === activeClassId);
                });
            }

            // 2. Update Dropdown
            const select = document.getElementById('annoClassSelect');
            if (select && select.value !== String(activeClassId)) {
                select.value = activeClassId;
            }

            // 3. Update Tick box / radio group
            const tickGroup = document.getElementById('annoClassTickGroup');
            if (tickGroup) {
                tickGroup.querySelectorAll('.anno-class-tick-label').forEach(lbl => {
                    const radio = lbl.querySelector('input[type="radio"]');
                    if (radio) {
                        const isMatch = parseInt(radio.value) === activeClassId;
                        radio.checked = isMatch;
                        lbl.classList.toggle('active', isMatch);
                    }
                });
            }
        }

        function renderClassChips() {
            // 1. Left sidebar chips
            const container = document.getElementById('classChipsContainer');
            if (container) {
                container.innerHTML = '';
                studioClasses.forEach(cls => {
                    const chip = document.createElement('div');
                    chip.className = 'class-chip ' + (cls.id === activeClassId ? 'active' : '');
                    chip.setAttribute('data-class-id', cls.id);
                    chip.innerHTML = `
                        <span class="color-dot" style="background:${cls.color};"></span>
                        <span>${cls.name} (${cls.id})</span>
                        ${cls.id >= 2 ? `<span class="chip-del-btn" onclick="event.stopPropagation(); deleteClass(${cls.id})">×</span>` : ''}
                    `;
                    chip.onclick = () => setActiveClass(cls.id);
                    container.appendChild(chip);
                });
            }

            // 2. Dropdown Selector
            const select = document.getElementById('annoClassSelect');
            if (select) {
                select.innerHTML = '';
                studioClasses.forEach(cls => {
                    const opt = document.createElement('option');
                    opt.value = cls.id;
                    opt.textContent = `${cls.name} (ID: ${cls.id})`;
                    if (cls.id === activeClassId) opt.selected = true;
                    select.appendChild(opt);
                });
            }

            // 3. Tick Boxes / Radio Group
            const tickGroup = document.getElementById('annoClassTickGroup');
            if (tickGroup) {
                tickGroup.innerHTML = '';
                studioClasses.forEach(cls => {
                    const lbl = document.createElement('label');
                    lbl.className = 'anno-class-tick-label ' + (cls.id === activeClassId ? 'active' : '');
                    lbl.title = `Switch active wire class to ${cls.name}`;
                    lbl.innerHTML = `
                        <input type="radio" name="annoClassRadio" class="anno-class-tick-input" value="${cls.id}" ${cls.id === activeClassId ? 'checked' : ''} onchange="setActiveClass(this.value)">
                        <span class="color-dot" style="background:${cls.color}; width:8px; height:8px;"></span>
                        <span>${cls.name}</span>
                    `;
                    tickGroup.appendChild(lbl);
                });
            }

            updateClassSelectors();
        }

        function openAddClassModal() {
            document.getElementById('addClassModal').classList.remove('hidden');
        }

        function closeAddClassModal() {
            document.getElementById('addClassModal').classList.add('hidden');
        }

        async function submitAddClass() {
            const name = document.getElementById('newClassNameInput').value.trim();
            const color = document.getElementById('newClassColorInput').value;
            if (!name) return alert("Please enter a class name");

            try {
                const res = await fetch('/api/classes/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ name, color })
                });
                const data = await res.json();
                if (data.success) {
                    closeAddClassModal();
                    document.getElementById('newClassNameInput').value = '';
                    await fetchClasses();
                } else {
                    alert(data.error || "Failed to add class");
                }
            } catch (e) {
                console.error(e);
            }
        }

        async function deleteClass(clsId) {
            if (!confirm(`Are you sure you want to remove class ID ${clsId}?`)) return;
            try {
                const res = await fetch('/api/classes/delete', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ id: clsId })
                });
                const data = await res.json();
                if (data.success) {
                    activeClassId = 0;
                    await fetchClasses();
                }
            } catch (e) {
                console.error(e);
            }
        }

        async function fetchStudioImages() {
            try {
                const res = await fetch('/api/annotation/images');
                const data = await res.json();
                const container = document.getElementById('studioImageList');
                container.innerHTML = '';
                document.getElementById('dsTotalImages').textContent = data.images.length;

                data.images.forEach(img => {
                    const row = document.createElement('div');
                    row.className = 'image-item-row ' + (img.name === currentImageName ? 'active' : '');
                    row.innerHTML = `
                        <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:130px;" title="${img.name}">${img.name}</span>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <span style="font-size:0.68rem; font-weight:700; color:${img.labeled ? '#34d399' : '#94a3b8'};">
                                ${img.labeled ? `✔ ${img.wires_count}` : 'Unlabeled'}
                            </span>
                            <button class="tool-btn" style="padding:2px 5px; font-size:0.68rem; color:#f87171; background:transparent; border-color:transparent;" title="Delete Image" onclick="event.stopPropagation(); deleteImage('${img.name}')">🗑️</button>
                        </div>
                    `;
                    row.onclick = () => loadStudioImage(img.name);
                    container.appendChild(row);
                });

                if (!currentImageName && data.images.length > 0) {
                    loadStudioImage(data.images[0].name);
                }
            } catch (err) {
                console.error("Failed to load studio images:", err);
            }
        }

        async function loadStudioImage(imgName) {
            currentImageName = imgName;
            document.getElementById('currentImgBadge').textContent = imgName;
            try {
                const res = await fetch('/api/annotation/image_data?file=' + encodeURIComponent(imgName));
                const data = await res.json();
                if (data.success) {
                    const img = new Image();
                    img.onload = () => {
                        currentImageLoaded = img;
                        imgOrigW = data.width || img.naturalWidth;
                        imgOrigH = data.height || img.naturalHeight;
                        annoCanvas.width = imgOrigW;
                        annoCanvas.height = imgOrigH;
                        annoWires = data.wires || [];
                        pendingPoints = [];
                        updateStepNotice();
                        redrawCanvas();
                        renderFinishedWires();
                    };
                    img.src = data.image_base64;
                    fetchStudioImages(); // refresh active highlight
                }
            } catch (e) {
                console.error(e);
            }
        }

        function handleStudioUpload(e) {
            const files = Array.from(e.target.files);
            if (!files.length) return;
            const formData = new FormData();
            files.forEach(f => formData.append('files', f));

            fetch('/api/annotation/upload', {
                method: 'POST',
                body: formData
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    fetchStudioImages();
                    if (data.first_image) loadStudioImage(data.first_image);
                }
            });
        }

        async function snapCameraToAnnotate() {
            if (!webcam.videoWidth) {
                alert("Please start the camera in the Inspection Station first.");
                return;
            }
            const canvas = document.getElementById('captureCanvas');
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
            const b64 = canvas.toDataURL('image/jpeg', 0.92);

            const name = `capture_${Date.now()}.jpg`;
            const res = await fetch('/api/annotation/upload_base64', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name, image_base64: b64 })
            });
            const data = await res.json();
            if (data.success) {
                await fetchStudioImages();
                loadStudioImage(name);
            }
        }

        // ================= CANVAS ANNOTATION INTERACTION =================
        annoCanvas.addEventListener('click', (e) => {
            if (!currentImageLoaded) return;
            const rect = annoCanvas.getBoundingClientRect();
            const scaleX = annoCanvas.width / rect.width;
            const scaleY = annoCanvas.height / rect.height;
            const x = Math.round((e.clientX - rect.left) * scaleX);
            const y = Math.round((e.clientY - rect.top) * scaleY);

            pendingPoints.push([x, y]);

            if (pendingPoints.length === 3) {
                // Wire completed!
                annoWires.push({
                    class_id: activeClassId,
                    points: [...pendingPoints]
                });
                pendingPoints = [];
                renderFinishedWires();
            }

            updateStepNotice();
            redrawCanvas();
        });

        function updateStepNotice() {
            const notice = document.getElementById('canvasStepNotice');
            const activeClassObj = studioClasses.find(c => c.id === activeClassId) || { name: 'Wire' };
            if (pendingPoints.length === 0) {
                notice.textContent = `Click 1: Place ${activeClassObj.name} ORIGIN (Coil)`;
            } else if (pendingPoints.length === 1) {
                notice.textContent = `Click 2: Place ${activeClassObj.name} MIDPOINT (Grommet)`;
            } else if (pendingPoints.length === 2) {
                notice.textContent = `Click 3: Place ${activeClassObj.name} TERMINAL (Socket)`;
            }
        }

        function undoPoint() {
            if (pendingPoints.length > 0) {
                pendingPoints.pop();
            } else if (annoWires.length > 0) {
                annoWires.pop();
                renderFinishedWires();
            }
            updateStepNotice();
            redrawCanvas();
        }

        function clearCurrentWire() {
            pendingPoints = [];
            updateStepNotice();
            redrawCanvas();
        }

        function deleteWireByIndex(idx) {
            annoWires.splice(idx, 1);
            renderFinishedWires();
            redrawCanvas();
        }

        function redrawCanvas() {
            if (!currentImageLoaded) return;
            ctxAnno.clearRect(0, 0, annoCanvas.width, annoCanvas.height);
            ctxAnno.drawImage(currentImageLoaded, 0, 0, annoCanvas.width, annoCanvas.height);

            // Draw Completed Wires
            annoWires.forEach((wire, widx) => {
                const clsObj = studioClasses.find(c => c.id === wire.class_id) || { color: '#ef4444', name: 'wire' };
                const color = clsObj.color || '#ef4444';
                const pts = wire.points;

                // Draw bounding box
                const xs = pts.map(p => p[0]);
                const ys = pts.map(p => p[1]);
                const bx1 = Math.max(0, Math.min(...xs) - 20);
                const by1 = Math.max(0, Math.min(...ys) - 20);
                const bw = Math.min(annoCanvas.width, Math.max(...xs) + 20) - bx1;
                const bh = Math.min(annoCanvas.height, Math.max(...ys) + 20) - by1;

                ctxAnno.strokeStyle = color;
                ctxAnno.lineWidth = 1.5;
                ctxAnno.setLineDash([4, 4]);
                ctxAnno.strokeRect(bx1, by1, bw, bh);
                ctxAnno.setLineDash([]);

                // Draw Class Badge
                ctxAnno.fillStyle = color;
                ctxAnno.fillRect(bx1, Math.max(0, by1 - 18), 85, 18);
                ctxAnno.fillStyle = "#ffffff";
                ctxAnno.font = "bold 11px sans-serif";
                ctxAnno.fillText(clsObj.name, bx1 + 4, Math.max(12, by1 - 5));

                // Draw Connecting Lines
                ctxAnno.beginPath();
                ctxAnno.moveTo(pts[0][0], pts[0][1]);
                ctxAnno.lineTo(pts[1][0], pts[1][1]);
                ctxAnno.lineTo(pts[2][0], pts[2][1]);
                ctxAnno.strokeStyle = color;
                ctxAnno.lineWidth = 3;
                ctxAnno.stroke();

                // Draw Keypoints
                pts.forEach((pt, pidx) => {
                    const label = pidx === 0 ? "Orig" : (pidx === 1 ? "Mid" : "Term");
                    ctxAnno.beginPath();
                    ctxAnno.arc(pt[0], pt[1], 7, 0, 2 * Math.PI);
                    ctxAnno.fillStyle = color;
                    ctxAnno.fill();
                    ctxAnno.strokeStyle = "#ffffff";
                    ctxAnno.lineWidth = 2;
                    ctxAnno.stroke();

                    ctxAnno.fillStyle = "#ffffff";
                    ctxAnno.font = "bold 12px sans-serif";
                    ctxAnno.fillText(`${pidx+1}:${label}`, pt[0] + 10, pt[1] + 4);
                });
            });

            // Draw Pending Points
            if (pendingPoints.length > 0) {
                const clsObj = studioClasses.find(c => c.id === activeClassId) || { color: '#3b82f6' };
                const color = clsObj.color;

                for (let i = 0; i < pendingPoints.length - 1; i++) {
                    ctxAnno.beginPath();
                    ctxAnno.moveTo(pendingPoints[i][0], pendingPoints[i][1]);
                    ctxAnno.lineTo(pendingPoints[i+1][0], pendingPoints[i+1][1]);
                    ctxAnno.strokeStyle = color;
                    ctxAnno.lineWidth = 2;
                    ctxAnno.stroke();
                }

                pendingPoints.forEach((pt, pidx) => {
                    ctxAnno.beginPath();
                    ctxAnno.arc(pt[0], pt[1], 8, 0, 2 * Math.PI);
                    ctxAnno.fillStyle = color;
                    ctxAnno.fill();
                    ctxAnno.strokeStyle = "#ffffff";
                    ctxAnno.lineWidth = 2;
                    ctxAnno.stroke();
                    ctxAnno.fillStyle = "#ffffff";
                    ctxAnno.fillText(`${pidx+1}`, pt[0] + 10, pt[1] - 4);
                });
            }
        }

        let isWiresCollapsed = false;

        function toggleWiresCollapse() {
            isWiresCollapsed = !isWiresCollapsed;
            const box = document.getElementById('finishedWiresSection');
            const btn = document.getElementById('btnToggleWiresList');
            if (!box || !btn) return;
            if (isWiresCollapsed) {
                box.classList.add('collapsed');
                btn.textContent = '▲ Show Wires';
            } else {
                box.classList.remove('collapsed');
                btn.textContent = '▼ Hide Wires';
            }
            setTimeout(redrawCanvas, 60);
        }

        function renderFinishedWires() {
            const list = document.getElementById('finishedWiresList');
            const badge = document.getElementById('finishedWiresBadge');
            if (badge) badge.textContent = annoWires.length;

            if (annoWires.length === 0) {
                list.innerHTML = '<span style="color:#64748b; font-size:0.75rem;">No wires labeled yet. Click 3 points per wire.</span>';
                return;
            }
            list.innerHTML = '';
            annoWires.forEach((w, idx) => {
                const cls = studioClasses.find(c => c.id === w.class_id) || { name: 'wire', color: '#ef4444' };
                const p0 = w.points[0];
                const p2 = w.points[2];
                const item = document.createElement('div');
                item.className = 'wire-item-pill';

                let classOpts = '';
                studioClasses.forEach(c => {
                    classOpts += `<option value="${c.id}" ${c.id === w.class_id ? 'selected' : ''}>${c.name}</option>`;
                });

                item.innerHTML = `
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="color-dot" style="background:${cls.color}; width:8px; height:8px;"></span>
                        <select onchange="changeWireClass(${idx}, this.value)" title="Change wire class" style="padding:1px 6px; font-size:0.72rem; font-weight:600; border-radius:4px; background:#1e293b; color:#fff; border:1px solid #475569;">
                            ${classOpts}
                        </select>
                        <span style="color:#94a3b8; font-size:0.7rem;">Orig:(${p0[0]},${p0[1]}) ➔ Term:(${p2[0]},${p2[1]})</span>
                    </div>
                    <button class="tool-btn" style="padding:2px 6px; font-size:0.7rem; color:#f87171;" onclick="deleteWireByIndex(${idx})">Delete</button>
                `;
                list.appendChild(item);
            });
        }

        function changeWireClass(idx, newCid) {
            if (annoWires[idx]) {
                annoWires[idx].class_id = parseInt(newCid);
                renderFinishedWires();
                redrawCanvas();
            }
        }

        async function saveAnnotation() {
            if (!currentImageName) return alert("No image loaded to save.");
            if (annoWires.length === 0) {
                if (!confirm("Save this image with 0 wires (empty/negative sample)?")) return;
            }

            try {
                const res = await fetch('/api/annotation/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image_name: currentImageName,
                        width: imgOrigW,
                        height: imgOrigH,
                        wires: annoWires
                    })
                });
                const data = await res.json();
                if (data.success) {
                    alert("✔ Annotation saved successfully!");
                    fetchStudioImages();
                } else {
                    alert("Save error: " + data.error);
                }
            } catch (e) {
                console.error(e);
            }
        }

        // ================= FULLSCREEN & DELETE IMAGE =================
        let isFullscreenAnno = false;

        function toggleFullscreenAnnotation() {
            const card = document.getElementById('annoCenterCard');
            isFullscreenAnno = !isFullscreenAnno;

            if (isFullscreenAnno) {
                card.classList.add('canvas-card-fullscreen');
                document.getElementById('btnToggleFs').innerHTML = '✕ Exit Fullscreen';
                document.getElementById('btnToggleFs').style.background = '#e11d48';
                if (document.documentElement.requestFullscreen) {
                    document.documentElement.requestFullscreen().catch(() => {});
                }
            } else {
                card.classList.remove('canvas-card-fullscreen');
                document.getElementById('btnToggleFs').innerHTML = '⛶ Fullscreen';
                document.getElementById('btnToggleFs').style.background = '#273549';
                if (document.exitFullscreen && document.fullscreenElement) {
                    document.exitFullscreen().catch(() => {});
                }
            }

            setTimeout(() => {
                redrawCanvas();
            }, 80);
        }

        document.addEventListener('keydown', (e) => {
            if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

            if (e.key === 'Escape' && isFullscreenAnno) {
                toggleFullscreenAnnotation();
                return;
            }

            // Quick class hotkeys in Studio: R (Red), B (Black), or 1..9
            const studioView = document.getElementById('viewStudio');
            if (studioView && studioView.classList.contains('active')) {
                if (e.key === 'r' || e.key === 'R') {
                    const c = studioClasses.find(x => x.name.toLowerCase().includes('red') || x.id === 0);
                    if (c) setActiveClass(c.id);
                } else if (e.key === 'b' || e.key === 'B') {
                    const c = studioClasses.find(x => x.name.toLowerCase().includes('black') || x.id === 1);
                    if (c) setActiveClass(c.id);
                } else if (e.key >= '1' && e.key <= '9') {
                    const idx = parseInt(e.key) - 1;
                    if (idx < studioClasses.length) {
                        setActiveClass(studioClasses[idx].id);
                    }
                }
            }
        });

        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement && isFullscreenAnno) {
                toggleFullscreenAnnotation();
            }
        });

        window.addEventListener('resize', () => {
            if (currentImageLoaded && document.getElementById('viewStudio') && document.getElementById('viewStudio').classList.contains('active')) {
                redrawCanvas();
            }
        });

        async function deleteImage(imgName) {
            if (!confirm(`Are you sure you want to permanently delete '${imgName}' and its annotations?`)) {
                return;
            }
            try {
                const res = await fetch('/api/annotation/delete_image', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image_name: imgName })
                });
                const data = await res.json();
                if (data.success) {
                    if (currentImageName === imgName) {
                        currentImageName = '';
                        currentImageLoaded = null;
                        annoWires = [];
                        pendingPoints = [];
                        ctxAnno.clearRect(0, 0, annoCanvas.width, annoCanvas.height);
                        document.getElementById('finishedWiresList').innerHTML = '<span style="color:#94a3b8; font-size:0.75rem;">Image deleted. Select another image.</span>';
                        document.getElementById('currentImgBadge').textContent = 'No image loaded';
                    }
                    await fetchStudioImages();
                } else {
                    alert("Delete failed: " + (data.error || "Unknown error"));
                }
            } catch (err) {
                console.error("Delete error:", err);
            }
        }

        function deleteCurrentImage() {
            if (!currentImageName) {
                alert("No image is currently loaded to delete.");
                return;
            }
            deleteImage(currentImageName);
        }

        // ================= RETRAINING MANAGEMENT =================
        async function startRetraining() {
            const epochs = parseInt(document.getElementById('trainEpochsInput').value) || 15;
            const batch = parseInt(document.getElementById('trainBatchInput').value) || 8;
            const weights = document.getElementById('trainWeightsSelect').value;

            if (!confirm(`Start retraining YOLOv8-pose with ${epochs} epochs and batch size ${batch}?`)) return;

            document.getElementById('btnStartTrain').disabled = true;
            document.getElementById('btnStartTrain').textContent = "Training In Progress...";

            try {
                const res = await fetch('/api/train/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ epochs, batch_size: batch, weights_choice: weights })
                });
                const data = await res.json();
                if (data.success) {
                    pollTrainingStatus();
                } else {
                    alert(data.error || "Could not start training");
                    document.getElementById('btnStartTrain').disabled = false;
                    document.getElementById('btnStartTrain').textContent = "🚀 Retrain Model";
                }
            } catch (e) {
                console.error(e);
            }
        }

        function pollTrainingStatus() {
            const interval = setInterval(async () => {
                try {
                    const res = await fetch('/api/train/status');
                    const data = await res.json();
                    
                    document.getElementById('trainStatusBadge').textContent = data.status.toUpperCase();
                    document.getElementById('trainProgressBar').style.width = data.progress_percent + '%';
                    document.getElementById('trainProgressText').textContent = data.progress_percent + '%';

                    const consoleBox = document.getElementById('trainConsole');
                    consoleBox.textContent = data.logs.join('\\n');
                    consoleBox.scrollTop = consoleBox.scrollHeight;

                    if (data.status === 'completed' || data.status === 'error' || data.status === 'idle') {
                        clearInterval(interval);
                        document.getElementById('btnStartTrain').disabled = false;
                        document.getElementById('btnStartTrain').textContent = "🚀 Retrain Model";
                        if (data.status === 'completed') {
                            document.getElementById('modelNameText').textContent = "best.pt (Retrained)";
                        }
                    }
                } catch (e) {
                    clearInterval(interval);
                }
            }, 1500);
        }

        // ================= SIDE MODEL (ORIGIN 3) STUDIO LOGIC =================
        let sideCurrentImgName = "";
        let sideCurrentImgLoaded = null;
        let sideImgOrigW = 1280, sideImgOrigH = 720;
        let sideBBox = null; // [x1, y1, x2, y2]
        let sideIsDrawing = false;
        let sideDrawStartX = 0, sideDrawStartY = 0, sideDrawCurX = 0, sideDrawCurY = 0;
        let sideImagesList = [];
        let sideCurrentFilter = 'all';
        let isFullscreenSideAnno = false;

        const sideCanvas = document.getElementById('sideCanvas');
        const ctxSide = sideCanvas ? sideCanvas.getContext('2d') : null;

        async function loadSideStudioData() {
            await fetchSideStats();
            await fetchSideImages();
        }

        async function fetchSideStats() {
            try {
                const res = await fetch('/api/origin3/dataset/stats');
                const data = await res.json();
                const elTot = document.getElementById('sideDsTotal');
                const elLab = document.getElementById('sideDsLabeled');
                const elUnlab = document.getElementById('sideDsUnlabeled');
                const elDev = document.getElementById('sideDsDevice');
                if (elTot) elTot.textContent = data.total;
                if (elLab) elLab.textContent = data.labeled;
                if (elUnlab) elUnlab.textContent = data.unlabeled;
                if (elDev) elDev.textContent = data.device;
            } catch (e) {
                console.error("fetchSideStats error:", e);
            }
        }

        async function fetchSideImages() {
            try {
                const res = await fetch('/api/origin3/annotation/images');
                const data = await res.json();
                sideImagesList = data.images || [];

                const countAll = sideImagesList.length;
                const countLab = sideImagesList.filter(i => i.labeled).length;
                const countUnlab = countAll - countLab;

                const elAll = document.getElementById('sideCountAll');
                const elLab = document.getElementById('sideCountLabeled');
                const elUnlab = document.getElementById('sideCountUnlabeled');
                if (elAll) elAll.textContent = countAll;
                if (elLab) elLab.textContent = countLab;
                if (elUnlab) elUnlab.textContent = countUnlab;

                renderSideImagesList();

                if (!sideCurrentImgName && sideImagesList.length > 0) {
                    loadSideImage(sideImagesList[0].name);
                }
            } catch (err) {
                console.error("fetchSideImages error:", err);
            }
        }

        function setSideFilter(filter) {
            sideCurrentFilter = filter;
            ['all', 'labeled', 'unlabeled'].forEach(f => {
                const btn = document.getElementById('sideFilterBtn' + f.charAt(0).toUpperCase() + f.slice(1));
                if (btn) btn.classList.toggle('active', f === filter);
            });
            renderSideImagesList();
        }

        function renderSideImagesList() {
            const container = document.getElementById('sideImageList');
            if (!container) return;
            container.innerHTML = '';

            let filtered = sideImagesList;
            if (sideCurrentFilter === 'labeled') {
                filtered = sideImagesList.filter(i => i.labeled);
            } else if (sideCurrentFilter === 'unlabeled') {
                filtered = sideImagesList.filter(i => !i.labeled);
            }

            if (filtered.length === 0) {
                container.innerHTML = '<div style="color:#64748b; font-size:0.75rem; padding:8px; text-align:center;">No images found in filter</div>';
                return;
            }

            filtered.forEach(img => {
                const row = document.createElement('div');
                row.className = 'image-item-row ' + (img.name === sideCurrentImgName ? 'active' : '');
                if (img.name === sideCurrentImgName) {
                    row.style.borderColor = '#0284c7';
                    row.style.background = 'rgba(2, 132, 199, 0.15)';
                }
                row.innerHTML = `
                    <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:130px; font-size:0.75rem;" title="${img.name}">${img.name}</span>
                    <div style="display:flex; align-items:center; gap:6px;">
                        <span style="font-size:0.68rem; font-weight:700; color:${img.labeled ? '#34d399' : '#f59e0b'};">
                            ${img.labeled ? '✔ Box' : '○ Empty'}
                        </span>
                        <button class="tool-btn" style="padding:2px 5px; font-size:0.68rem; color:#f87171; background:transparent; border-color:transparent;" title="Delete Image" onclick="event.stopPropagation(); deleteSideImage('${img.name}')">🗑️</button>
                    </div>
                `;
                row.onclick = () => loadSideImage(img.name);
                container.appendChild(row);
            });
        }

        async function loadSideImage(imgName) {
            sideCurrentImgName = imgName;
            const badge = document.getElementById('sideCurrentImgBadge');
            if (badge) badge.textContent = imgName;

            try {
                const res = await fetch('/api/origin3/annotation/image_data?file=' + encodeURIComponent(imgName));
                const data = await res.json();
                if (data.success) {
                    const img = new Image();
                    img.onload = () => {
                        sideCurrentImgLoaded = img;
                        sideImgOrigW = data.width || img.naturalWidth;
                        sideImgOrigH = data.height || img.naturalHeight;
                        if (sideCanvas) {
                            sideCanvas.width = sideImgOrigW;
                            sideCanvas.height = sideImgOrigH;
                        }
                        sideBBox = data.bbox || null;
                        if (badge) badge.textContent = `${imgName} (${sideImgOrigW}×${sideImgOrigH})`;
                        updateSideBoxDetails();
                        redrawSideCanvas();
                    };
                    img.src = data.image_base64;
                    renderSideImagesList();
                }
            } catch (e) {
                console.error("loadSideImage error:", e);
            }
        }

        function updateSideBoxDetails() {
            const badge = document.getElementById('sideBoxBadge');
            const coords = document.getElementById('sideBoxCoords');
            if (!sideBBox) {
                if (badge) {
                    badge.textContent = 'None';
                    badge.style.color = '#94a3b8';
                    badge.style.borderColor = 'rgba(148, 163, 184, 0.4)';
                    badge.style.background = 'rgba(148, 163, 184, 0.1)';
                }
                if (coords) coords.textContent = 'No box defined yet. Drag across Origin 3 on canvas.';
            } else {
                const [x1, y1, x2, y2] = sideBBox;
                const w = x2 - x1;
                const h = y2 - y1;
                const cx = Math.round(x1 + w / 2);
                const cy = Math.round(y1 + h / 2);
                if (badge) {
                    badge.textContent = '1 Box Defined';
                    badge.style.color = '#38bdf8';
                    badge.style.borderColor = 'rgba(6, 182, 212, 0.4)';
                    badge.style.background = 'rgba(6, 182, 212, 0.2)';
                }
                if (coords) {
                    coords.innerHTML = `<b>origin_3</b> [${x1}, ${y1}, ${x2}, ${y2}] &bull; Center: (${cx}, ${cy}) &bull; Size: ${w}×${h} px`;
                }
            }
        }

        function redrawSideCanvas() {
            if (!sideCanvas || !ctxSide || !sideCurrentImgLoaded) return;
            ctxSide.clearRect(0, 0, sideCanvas.width, sideCanvas.height);
            ctxSide.drawImage(sideCurrentImgLoaded, 0, 0, sideCanvas.width, sideCanvas.height);

            // Draw dragging preview
            if (sideIsDrawing) {
                const x1 = Math.min(sideDrawStartX, sideDrawCurX);
                const y1 = Math.min(sideDrawStartY, sideDrawCurY);
                const w = Math.max(sideDrawStartX, sideDrawCurX) - x1;
                const h = Math.max(sideDrawStartY, sideDrawCurY) - y1;

                ctxSide.fillStyle = 'rgba(6, 182, 212, 0.18)';
                ctxSide.fillRect(x1, y1, w, h);
                ctxSide.strokeStyle = '#06b6d4';
                ctxSide.lineWidth = 2.5;
                ctxSide.setLineDash([6, 4]);
                ctxSide.strokeRect(x1, y1, w, h);
                ctxSide.setLineDash([]);
                return;
            }

            // Draw finalized box
            if (sideBBox) {
                const [x1, y1, x2, y2] = sideBBox;
                const w = x2 - x1;
                const h = y2 - y1;
                const cx = Math.round(x1 + w / 2);
                const cy = Math.round(y1 + h / 2);

                ctxSide.fillStyle = 'rgba(6, 182, 212, 0.22)';
                ctxSide.fillRect(x1, y1, w, h);

                ctxSide.strokeStyle = '#06b6d4';
                ctxSide.lineWidth = 3;
                ctxSide.strokeRect(x1, y1, w, h);

                ctxSide.strokeStyle = '#facc15';
                ctxSide.lineWidth = 2;
                ctxSide.beginPath();
                ctxSide.moveTo(cx - 10, cy);
                ctxSide.lineTo(cx + 10, cy);
                ctxSide.moveTo(cx, cy - 10);
                ctxSide.lineTo(cx, cy + 10);
                ctxSide.stroke();

                const gripSize = 8;
                ctxSide.fillStyle = '#ffffff';
                ctxSide.strokeStyle = '#06b6d4';
                ctxSide.lineWidth = 1.5;
                const grips = [
                    [x1, y1], [x2, y1], [x1, y2], [x2, y2],
                    [cx, y1], [cx, y2], [x1, cy], [x2, cy]
                ];
                grips.forEach(([gx, gy]) => {
                    ctxSide.fillRect(gx - gripSize / 2, gy - gripSize / 2, gripSize, gripSize);
                    ctxSide.strokeRect(gx - gripSize / 2, gy - gripSize / 2, gripSize, gripSize);
                });

                const labelText = "origin_3";
                ctxSide.font = "bold 13px sans-serif";
                const textWidth = ctxSide.measureText(labelText).width;
                const badgeHeight = 22;
                const badgeWidth = textWidth + 16;
                const badgeY = Math.max(0, y1 - badgeHeight);

                ctxSide.fillStyle = "#06b6d4";
                ctxSide.fillRect(x1, badgeY, badgeWidth, badgeHeight);
                ctxSide.fillStyle = "#04151f";
                ctxSide.fillText(labelText, x1 + 8, badgeY + 16);
            }
        }

        if (sideCanvas) {
            sideCanvas.addEventListener('mousedown', (e) => {
                if (!sideCurrentImgLoaded) return;
                const rect = sideCanvas.getBoundingClientRect();
                const scaleX = sideCanvas.width / rect.width;
                const scaleY = sideCanvas.height / rect.height;

                sideDrawStartX = Math.max(0, Math.min(sideCanvas.width, Math.round((e.clientX - rect.left) * scaleX)));
                sideDrawStartY = Math.max(0, Math.min(sideCanvas.height, Math.round((e.clientY - rect.top) * scaleY)));
                sideDrawCurX = sideDrawStartX;
                sideDrawCurY = sideDrawStartY;
                sideIsDrawing = true;
            });

            window.addEventListener('mousemove', (e) => {
                if (!sideIsDrawing || !sideCanvas) return;
                const rect = sideCanvas.getBoundingClientRect();
                const scaleX = sideCanvas.width / rect.width;
                const scaleY = sideCanvas.height / rect.height;

                sideDrawCurX = Math.max(0, Math.min(sideCanvas.width, Math.round((e.clientX - rect.left) * scaleX)));
                sideDrawCurY = Math.max(0, Math.min(sideCanvas.height, Math.round((e.clientY - rect.top) * scaleY)));
                redrawSideCanvas();
            });

            window.addEventListener('mouseup', (e) => {
                if (!sideIsDrawing) return;
                sideIsDrawing = false;
                const x1 = Math.min(sideDrawStartX, sideDrawCurX);
                const y1 = Math.min(sideDrawStartY, sideDrawCurY);
                const x2 = Math.max(sideDrawStartX, sideDrawCurX);
                const y2 = Math.max(sideDrawStartY, sideDrawCurY);

                if ((x2 - x1) > 8 && (y2 - y1) > 8) {
                    sideBBox = [x1, y1, x2, y2];
                    updateSideBoxDetails();
                }
                redrawSideCanvas();
            });
        }

        function clearSideBox() {
            sideBBox = null;
            updateSideBoxDetails();
            redrawSideCanvas();
        }

        async function sidePreAnnotate() {
            if (!sideCurrentImgName) return alert("Please select an image first.");
            const statusEl = document.getElementById('sideCanvasStatus');
            if (statusEl) statusEl.textContent = "🤖 Running Origin 3 Auto-Detection...";
            try {
                const res = await fetch('/api/origin3/annotation/pre_annotate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image_name: sideCurrentImgName })
                });
                const data = await res.json();
                if (data.success && data.bbox) {
                    sideBBox = data.bbox.map(v => Math.round(v));
                    updateSideBoxDetails();
                    redrawSideCanvas();
                    if (statusEl) statusEl.textContent = `Target: Origin 3 • Auto-Detected (${(data.confidence*100).toFixed(1)}%)`;
                    alert(`🤖 Origin 3 auto-detected with ${(data.confidence * 100).toFixed(1)}% confidence!\nClick 'Save Annotation' to confirm.`);
                } else {
                    if (statusEl) statusEl.textContent = "Target: Origin 3 • Drag to Draw Box";
                    alert(data.error || "Origin 3 not detected automatically. Please draw the box manually.");
                }
            } catch (err) {
                if (statusEl) statusEl.textContent = "Target: Origin 3 • Drag to Draw Box";
                alert("Auto-detect failed: " + err.message);
            }
        }

        async function saveSideAnnotation() {
            if (!sideCurrentImgName) return alert("No image loaded to save.");
            if (!sideBBox) {
                if (!confirm("Save this image with NO bounding box (empty label)?")) return;
            }

            try {
                const res = await fetch('/api/origin3/annotation/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image_name: sideCurrentImgName,
                        width: sideImgOrigW,
                        height: sideImgOrigH,
                        bbox: sideBBox
                    })
                });
                const data = await res.json();
                if (data.success) {
                    alert("✔ Origin 3 annotation saved successfully!");
                    await fetchSideImages();
                    await fetchSideStats();
                } else {
                    alert("Save error: " + data.error);
                }
            } catch (e) {
                console.error("saveSideAnnotation error:", e);
            }
        }

        function handleSideUpload(e) {
            const files = Array.from(e.target.files);
            if (!files.length) return;
            const formData = new FormData();
            files.forEach(f => formData.append('files', f));

            fetch('/api/origin3/annotation/upload', {
                method: 'POST',
                body: formData
            }).then(r => r.json()).then(async data => {
                if (data.success) {
                    await fetchSideImages();
                    await fetchSideStats();
                    if (data.first_image) loadSideImage(data.first_image);
                }
            }).catch(err => console.error("Side upload failed:", err));
        }

        async function snapSideCamera() {
            const cam = (webcam2 && webcam2.videoWidth) ? webcam2 : ((webcam1 && webcam1.videoWidth) ? webcam1 : null);
            if (!cam) {
                alert("Please start the cameras in the Inspection Station first.");
                return;
            }
            const canvas = document.getElementById('captureCanvas');
            canvas.width = cam.videoWidth;
            canvas.height = cam.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(cam, 0, 0, canvas.width, canvas.height);
            const b64 = canvas.toDataURL('image/jpeg', 0.92);

            const name = `side_snap_${Date.now()}.jpg`;
            const res = await fetch('/api/origin3/annotation/upload_base64', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name, image_base64: b64 })
            });
            const data = await res.json();
            if (data.success) {
                await fetchSideImages();
                await fetchSideStats();
                loadSideImage(name);
            }
        }

        async function deleteSideImage(imgName) {
            if (!confirm(`Are you sure you want to delete "${imgName}"?`)) return;
            try {
                const res = await fetch('/api/origin3/annotation/delete_image', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image_name: imgName })
                });
                const data = await res.json();
                if (data.success) {
                    if (sideCurrentImgName === imgName) {
                        sideCurrentImgName = "";
                        sideCurrentImgLoaded = null;
                        sideBBox = null;
                        if (sideCanvas && ctxSide) ctxSide.clearRect(0, 0, sideCanvas.width, sideCanvas.height);
                        updateSideBoxDetails();
                        const badge = document.getElementById('sideCurrentImgBadge');
                        if (badge) badge.textContent = 'No image loaded';
                    }
                    await fetchSideImages();
                    await fetchSideStats();
                } else {
                    alert("Delete failed: " + (data.error || "Unknown error"));
                }
            } catch (err) {
                console.error("Delete error:", err);
            }
        }

        function deleteCurrentSideImage() {
            if (!sideCurrentImgName) return alert("No image loaded to delete.");
            deleteSideImage(sideCurrentImgName);
        }

        function toggleFullscreenSideAnnotation() {
            const card = document.getElementById('sideCenterCard');
            isFullscreenSideAnno = !isFullscreenSideAnno;

            if (isFullscreenSideAnno) {
                card.classList.add('canvas-card-fullscreen');
                const btn = document.getElementById('btnToggleSideFs');
                if (btn) {
                    btn.innerHTML = '✕ Exit Fullscreen';
                    btn.style.background = '#e11d48';
                }
                if (document.documentElement.requestFullscreen) {
                    document.documentElement.requestFullscreen().catch(() => {});
                }
            } else {
                card.classList.remove('canvas-card-fullscreen');
                const btn = document.getElementById('btnToggleSideFs');
                if (btn) {
                    btn.innerHTML = '⛶ Fullscreen';
                    btn.style.background = '#273549';
                }
                if (document.exitFullscreen && document.fullscreenElement) {
                    document.exitFullscreen().catch(() => {});
                }
            }

            setTimeout(() => {
                redrawSideCanvas();
            }, 80);
        }

        async function startSideRetraining() {
            const epochs = parseInt(document.getElementById('sideTrainEpochsInput').value) || 30;
            const batch = parseInt(document.getElementById('sideTrainBatchInput').value) || 8;
            const weights = document.getElementById('sideTrainWeightsSelect').value;

            if (!confirm(`Start retraining YOLOv8 Origin 3 Side Model with ${epochs} epochs and batch size ${batch}?`)) return;

            const btn = document.getElementById('btnStartSideTrain');
            btn.disabled = true;
            btn.textContent = "Training In Progress...";

            try {
                const res = await fetch('/api/origin3/train/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ epochs, batch_size: batch, weights_choice: weights })
                });
                const data = await res.json();
                if (data.success) {
                    pollSideTrainingStatus();
                } else {
                    alert(data.error || "Could not start Origin 3 training");
                    btn.disabled = false;
                    btn.textContent = "🚀 Retrain Side Model (Origin 3)";
                }
            } catch (e) {
                console.error(e);
                btn.disabled = false;
                btn.textContent = "🚀 Retrain Side Model (Origin 3)";
            }
        }

        function pollSideTrainingStatus() {
            const interval = setInterval(async () => {
                try {
                    const res = await fetch('/api/origin3/train/status');
                    const data = await res.json();

                    const badge = document.getElementById('sideTrainStatusBadge');
                    if (badge) badge.textContent = data.status.toUpperCase();
                    const bar = document.getElementById('sideTrainProgressBar');
                    if (bar) bar.style.width = data.progress_percent + '%';
                    const txt = document.getElementById('sideTrainProgressText');
                    if (txt) txt.textContent = data.progress_percent + '%';

                    const consoleBox = document.getElementById('sideTrainConsole');
                    if (consoleBox && data.logs) {
                        consoleBox.textContent = data.logs.join('\\n');
                        consoleBox.scrollTop = consoleBox.scrollHeight;
                    }

                    if (data.status === 'completed' || data.status === 'error' || data.status === 'idle') {
                        clearInterval(interval);
                        const btn = document.getElementById('btnStartSideTrain');
                        if (btn) {
                            btn.disabled = false;
                            btn.textContent = "🚀 Retrain Side Model (Origin 3)";
                        }
                        if (data.status === 'completed') {
                            alert("🎉 Origin 3 Side Model retrained successfully!\\nUpdated weights hot-reloaded into the Dual Inspection Engine.");
                            fetchSideStats();
                        }
                    }
                } catch (e) {
                    clearInterval(interval);
                }
            }, 1500);
        }

        // Init
        window.addEventListener('DOMContentLoaded', () => {
            initCameras();
        });
    </script>
</body>
</html>
"""

# ================= SERVER ROUTES =================
@app.route("/")
def index():
    get_model()
    return render_template_string(HTML_TEMPLATE, model_name=MODEL_NAME)

@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    data = request.get_json(silent=True) or {}
    conf_pose = float(data.get("conf", 0.35))
    conf_orig3 = float(data.get("conf_orig3", 0.40))

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
            print("Base64 decode error:", e)
            return None

    top_img = None
    orig3_img = None

    # Top View Image (Pose)
    if "sample_name_top" in data and data["sample_name_top"]:
        s_path = os.path.join(SCRIPT_DIR, "test_samples", data["sample_name_top"])
        if os.path.exists(s_path):
            top_img = cv2.imread(s_path)
    elif "top_image_base64" in data and data["top_image_base64"]:
        top_img = decode_b64(data["top_image_base64"])
    elif "sample_name" in data and data["sample_name"]:
        s_path = os.path.join(SCRIPT_DIR, "test_samples", data["sample_name"])
        if os.path.exists(s_path):
            top_img = cv2.imread(s_path)
    elif "image_base64" in data and data["image_base64"]:
        top_img = decode_b64(data["image_base64"])

    # Origin 3 Image (Side View)
    if "sample_name_orig3" in data and data["sample_name_orig3"]:
        s_path = os.path.join(SCRIPT_DIR, "test_samples", "origin3", data["sample_name_orig3"])
        if not os.path.exists(s_path):
            s_path = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset", data["sample_name_orig3"])
        if os.path.exists(s_path):
            orig3_img = cv2.imread(s_path)
    elif "orig3_image_base64" in data and data["orig3_image_base64"]:
        orig3_img = decode_b64(data["orig3_image_base64"])

    # Intelligent fallback for single image test
    if top_img is None and orig3_img is not None:
        def_top = os.path.join(SCRIPT_DIR, "test_samples", "correctwiring (1).jpg")
        if os.path.exists(def_top):
            top_img = cv2.imread(def_top)
    elif orig3_img is None and top_img is not None:
        def_orig3 = os.path.join(SCRIPT_DIR, "test_samples", "origin3", "orig3_pass.jpg")
        if os.path.exists(def_orig3):
            orig3_img = cv2.imread(def_orig3)

    if top_img is None or orig3_img is None:
        return jsonify({"success": False, "error": "Both Top View (Pose) and Side View (Origin 3) images are required."}), 400

    dual_res = inspect_dual_system(top_img, orig3_img, conf_pose=conf_pose, conf_orig3=conf_orig3)
    parsed = parse_dual_result(dual_res)
    return jsonify(parsed)

# Class Management APIs
@app.route("/api/classes", methods=["GET"])
def api_get_classes():
    return jsonify(get_classes_data())

@app.route("/api/classes/add", methods=["POST"])
def api_add_class():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip().lower().replace(" ", "_")
    color = data.get("color", "#3b82f6")
    if not name:
        return jsonify({"success": False, "error": "Name required"}), 400

    with open(DATA_YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    names = {int(k): str(v) for k, v in names.items()}
    next_id = max(names.keys()) + 1 if names else 0
    names[next_id] = name
    cfg["names"] = names

    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    save_class_color(next_id, color)
    return jsonify({"success": True, "classes": get_classes_data()})

@app.route("/api/classes/delete", methods=["POST"])
def api_delete_class():
    data = request.get_json(silent=True) or {}
    cid = int(data.get("id", -1))
    if cid < 2:
        return jsonify({"success": False, "error": "Cannot delete core red/black classes"}), 400

    with open(DATA_YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    names = {int(k): str(v) for k, v in names.items()}
    if cid in names:
        del names[cid]
    cfg["names"] = names

    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    return jsonify({"success": True, "classes": get_classes_data()})

# Annotation APIs
@app.route("/api/annotation/images", methods=["GET"])
def api_anno_images():
    train_dir = os.path.join(SCRIPT_DIR, "dataset", "images", "train")
    val_dir = os.path.join(SCRIPT_DIR, "dataset", "images", "val")
    raw_dir = os.path.join(SCRIPT_DIR, "dataset", "images", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    images = []
    # Train
    if os.path.exists(train_dir):
        for f in os.listdir(train_dir):
            if f.lower().endswith((".jpg", ".png")):
                lbl = os.path.join(SCRIPT_DIR, "dataset", "labels", "train", os.path.splitext(f)[0] + ".txt")
                cnt = 0
                if os.path.exists(lbl):
                    with open(lbl) as fp: cnt = len(fp.readlines())
                images.append({"name": f, "split": "train", "labeled": cnt > 0, "wires_count": cnt})

    # Raw / Newly uploaded
    for f in os.listdir(raw_dir):
        if f.lower().endswith((".jpg", ".png")):
            images.append({"name": f, "split": "raw", "labeled": False, "wires_count": 0})

    return jsonify({"images": images})

@app.route("/api/annotation/upload", methods=["POST"])
def api_anno_upload():
    raw_dir = os.path.join(SCRIPT_DIR, "dataset", "images", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    files = request.files.getlist("files")
    first_name = ""
    for f in files:
        if f.filename:
            path = os.path.join(raw_dir, f.filename)
            f.save(path)
            if not first_name: first_name = f.filename
    return jsonify({"success": True, "first_image": first_name})

@app.route("/api/annotation/upload_base64", methods=["POST"])
def api_anno_upload_base64():
    raw_dir = os.path.join(SCRIPT_DIR, "dataset", "images", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    data = request.get_json(silent=True) or {}
    name = data.get("name", f"snap_{int(time.time())}.jpg")
    b64_str = data.get("image_base64", "")
    if "," in b64_str: b64_str = b64_str.split(",")[1]
    raw_bytes = base64.b64decode(b64_str)
    with open(os.path.join(raw_dir, name), "wb") as fp:
        fp.write(raw_bytes)
    return jsonify({"success": True, "name": name})

@app.route("/api/annotation/image_data", methods=["GET"])
def api_anno_image_data():
    img_name = request.args.get("file", "")
    if not img_name:
        return jsonify({"success": False, "error": "file param missing"}), 400

    # Locate image
    img_path = None
    for folder in ["raw", "train", "val"]:
        p = os.path.join(SCRIPT_DIR, "dataset", "images", folder, img_name)
        if os.path.exists(p):
            img_path = p
            break

    if not img_path:
        return jsonify({"success": False, "error": "Image file not found"}), 404

    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"success": False, "error": "Cannot load image"}), 400

    h, w = img.shape[:2]
    b64 = cv2_to_base64(img)

    # Search for existing label file
    wires = []
    for split in ["train", "val"]:
        lbl_p = os.path.join(SCRIPT_DIR, "dataset", "labels", split, os.path.splitext(img_name)[0] + ".txt")
        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as fp:
                for line in fp:
                    parts = line.strip().split()
                    if len(parts) >= 14:
                        cid = int(parts[0])
                        p1 = [round(float(parts[5]) * w), round(float(parts[6]) * h)]
                        p2 = [round(float(parts[8]) * w), round(float(parts[9]) * h)]
                        p3 = [round(float(parts[11]) * w), round(float(parts[12]) * h)]
                        wires.append({"class_id": cid, "points": [p1, p2, p3]})
            break

    return jsonify({"success": True, "image_base64": b64, "width": w, "height": h, "wires": wires})

@app.route("/api/annotation/save", methods=["POST"])
def api_anno_save():
    data = request.get_json(silent=True) or {}
    img_name = data.get("image_name", "")
    w = int(data.get("width", 1280))
    h = int(data.get("height", 720))
    wires = data.get("wires", [])

    # If file was in raw, move it to train
    raw_path = os.path.join(SCRIPT_DIR, "dataset", "images", "raw", img_name)
    target_split = "train"
    train_img_path = os.path.join(SCRIPT_DIR, "dataset", "images", "train", img_name)
    val_img_path = os.path.join(SCRIPT_DIR, "dataset", "images", "val", img_name)

    if os.path.exists(val_img_path):
        target_split = "val"
    elif os.path.exists(raw_path):
        os.makedirs(os.path.dirname(train_img_path), exist_ok=True)
        shutil.move(raw_path, train_img_path)

    # Format YOLO pose lines
    lbl_dir = os.path.join(SCRIPT_DIR, "dataset", "labels", target_split)
    os.makedirs(lbl_dir, exist_ok=True)
    lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

    lines = []
    pad = 20
    for wire in wires:
        cid = int(wire["class_id"])
        pts = wire["points"] # [[x1, y1], [x2, y2], [x3, y3]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bx1 = max(0, min(xs) - pad)
        by1 = max(0, min(ys) - pad)
        bx2 = min(w, max(xs) + pad)
        by2 = min(h, max(ys) + pad)

        bw = bx2 - bx1
        bh = by2 - by1
        bcx = bx1 + bw / 2.0
        bcy = by1 + bh / 2.0

        norm_bcx = bcx / w
        norm_bcy = bcy / h
        norm_bw = bw / w
        norm_bh = bh / h

        kpts_str = []
        for p in pts:
            kpts_str.append(f"{p[0] / w:.6f} {p[1] / h:.6f} 2")

        line = f"{cid} {norm_bcx:.6f} {norm_bcy:.6f} {norm_bw:.6f} {norm_bh:.6f} " + " ".join(kpts_str)
        lines.append(line)

    with open(lbl_path, "w") as fp:
        fp.write("\n".join(lines) + ("\n" if lines else ""))

    # Clear label cache
    for c_file in glob.glob(os.path.join(SCRIPT_DIR, "dataset", "labels", "*.cache")):
        try: os.remove(c_file)
        except Exception: pass

    return jsonify({"success": True, "saved_file": lbl_path, "wire_count": len(wires)})

@app.route("/api/annotation/delete_image", methods=["POST"])
def api_anno_delete_image():
    data = request.get_json(silent=True) or {}
    img_name = data.get("image_name", "").strip()
    if not img_name:
        return jsonify({"success": False, "error": "No image name provided"}), 400

    safe_name = os.path.basename(img_name)
    stem = os.path.splitext(safe_name)[0]

    deleted_any = False
    for sub in ["raw", "train", "val"]:
        img_p = os.path.join(SCRIPT_DIR, "dataset", "images", sub, safe_name)
        if os.path.exists(img_p):
            try:
                os.remove(img_p)
                deleted_any = True
            except Exception as e:
                return jsonify({"success": False, "error": f"Failed to delete image: {str(e)}"}), 500

    for sub in ["train", "val"]:
        lbl_p = os.path.join(SCRIPT_DIR, "dataset", "labels", sub, stem + ".txt")
        if os.path.exists(lbl_p):
            try:
                os.remove(lbl_p)
            except Exception:
                pass

    # Clear label cache so retraining picks up dataset changes
    for pattern in [
        os.path.join(SCRIPT_DIR, "dataset", "labels", "*.cache"),
        os.path.join(SCRIPT_DIR, "dataset", "labels", "*", "*.cache")
    ]:
        for c_file in glob.glob(pattern):
            try: os.remove(c_file)
            except Exception: pass

    return jsonify({"success": True, "deleted": safe_name, "found": deleted_any})

# Retraining APIs
@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    data = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 15))
    batch = int(data.get("batch_size", 8))
    weights = data.get("weights_choice", "best")

    success, msg = training_mgr.start(epochs=epochs, batch=batch, weights_choice=weights)
    if success:
        return jsonify({"success": True, "message": msg})
    return jsonify({"success": False, "error": msg}), 400

@app.route("/api/train/status", methods=["GET"])
def api_train_status():
    return jsonify({
        "status": training_mgr.status,
        "current_epoch": training_mgr.current_epoch,
        "total_epochs": training_mgr.total_epochs,
        "progress_percent": training_mgr.progress_percent,
        "logs": training_mgr.logs[-50:],
        "error": training_mgr.error_msg
    })

# ================= ORIGIN 3 SIDE MODEL ROUTES =================
@app.route("/api/origin3/annotation/images", methods=["GET"])
def api_origin3_images():
    images = []
    if os.path.exists(ORIGIN3_IMAGES_DIR):
        for f in sorted(os.listdir(ORIGIN3_IMAGES_DIR)):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                stem = os.path.splitext(f)[0]
                lbl_path = os.path.join(ORIGIN3_LABELS_DIR, f"{stem}.txt")
                labeled = False
                box_count = 0
                if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                    with open(lbl_path, "r", encoding="utf-8") as fp:
                        lines = [l.strip() for l in fp if l.strip()]
                        box_count = len(lines)
                        labeled = box_count > 0
                images.append({"name": f, "labeled": labeled, "box_count": box_count})
    return jsonify({"images": images})

@app.route("/api/origin3/annotation/upload", methods=["POST"])
def api_origin3_upload():
    files = request.files.getlist("files")
    first_name = ""
    for f in files:
        if f.filename:
            safe_name = os.path.basename(f.filename)
            path = os.path.join(ORIGIN3_IMAGES_DIR, safe_name)
            f.save(path)
            if not first_name:
                first_name = safe_name
    return jsonify({"success": True, "first_image": first_name})

@app.route("/api/origin3/annotation/upload_base64", methods=["POST"])
def api_origin3_upload_base64():
    data = request.get_json(silent=True) or {}
    name = data.get("name", f"snap_side_{int(time.time())}.jpg")
    safe_name = os.path.basename(name)
    b64_str = data.get("image_base64", "")
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    raw_bytes = base64.b64decode(b64_str)
    with open(os.path.join(ORIGIN3_IMAGES_DIR, safe_name), "wb") as fp:
        fp.write(raw_bytes)
    return jsonify({"success": True, "name": safe_name})

@app.route("/api/origin3/annotation/image_data", methods=["GET"])
def api_origin3_image_data():
    img_name = request.args.get("file", "")
    if not img_name:
        return jsonify({"success": False, "error": "file param missing"}), 400
    safe_name = os.path.basename(img_name)
    img_path = os.path.join(ORIGIN3_IMAGES_DIR, safe_name)
    if not os.path.exists(img_path):
        return jsonify({"success": False, "error": "Image file not found"}), 404

    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"success": False, "error": "Cannot load image"}), 400

    h, w = img.shape[:2]
    b64 = cv2_to_base64(img)

    bbox = None
    stem = os.path.splitext(safe_name)[0]
    lbl_path = os.path.join(ORIGIN3_LABELS_DIR, f"{stem}.txt")
    if os.path.exists(lbl_path):
        with open(lbl_path, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = max(0, int(round((xc - bw / 2.0) * w)))
                    y1 = max(0, int(round((yc - bh / 2.0) * h)))
                    x2 = min(w, int(round((xc + bw / 2.0) * w)))
                    y2 = min(h, int(round((yc + bh / 2.0) * h)))
                    bbox = [x1, y1, x2, y2]
                    break

    return jsonify({"success": True, "image_base64": b64, "width": w, "height": h, "bbox": bbox})

@app.route("/api/origin3/annotation/save", methods=["POST"])
def api_origin3_save():
    data = request.get_json(silent=True) or {}
    img_name = data.get("image_name", "")
    safe_name = os.path.basename(img_name)
    w = int(data.get("width", 1280))
    h = int(data.get("height", 720))
    bbox = data.get("bbox", None)

    stem = os.path.splitext(safe_name)[0]
    lbl_path = os.path.join(ORIGIN3_LABELS_DIR, f"{stem}.txt")

    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw = x2 - x1
        bh = y2 - y1
        if bw > 0 and bh > 0:
            xc = (x1 + x2) / (2.0 * w)
            yc = (y1 + y2) / (2.0 * h)
            norm_w = bw / float(w)
            norm_h = bh / float(h)
            with open(lbl_path, "w", encoding="utf-8") as fp:
                fp.write(f"0 {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            return jsonify({"success": True, "saved_file": lbl_path, "has_box": True})

    if os.path.exists(lbl_path):
        try:
            os.remove(lbl_path)
        except Exception:
            pass
    return jsonify({"success": True, "saved_file": lbl_path, "has_box": False})

@app.route("/api/origin3/annotation/delete_image", methods=["POST"])
def api_origin3_delete_image():
    data = request.get_json(silent=True) or {}
    img_name = data.get("image_name", "").strip()
    safe_name = os.path.basename(img_name)
    stem = os.path.splitext(safe_name)[0]

    img_p = os.path.join(ORIGIN3_IMAGES_DIR, safe_name)
    lbl_p = os.path.join(ORIGIN3_LABELS_DIR, f"{stem}.txt")

    deleted = False
    if os.path.exists(img_p):
        try:
            os.remove(img_p)
            deleted = True
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    if os.path.exists(lbl_p):
        try:
            os.remove(lbl_p)
        except Exception:
            pass

    return jsonify({"success": True, "deleted": safe_name, "found": deleted})

@app.route("/api/origin3/annotation/pre_annotate", methods=["POST"])
def api_origin3_pre_annotate():
    data = request.get_json(silent=True) or {}
    img_name = data.get("image_name", "")
    safe_name = os.path.basename(img_name)
    img_path = os.path.join(ORIGIN3_IMAGES_DIR, safe_name)
    if not os.path.exists(img_path):
        return jsonify({"success": False, "error": "Image file not found"}), 404

    try:
        model = get_origin3_model()
        img = cv2.imread(img_path)
        if img is None:
            return jsonify({"success": False, "error": "Cannot read image"}), 400

        res = inspect_origin3(model, img, conf=0.25)
        if res.get("is_pass") and res.get("bbox"):
            return jsonify({
                "success": True,
                "bbox": res["bbox"],
                "confidence": res.get("confidence", 0.0),
                "center": res.get("center", None)
            })
        else:
            return jsonify({
                "success": False,
                "error": "Origin 3 not detected with confidence threshold (>= 0.25). Please annotate manually."
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/origin3/dataset/stats", methods=["GET"])
def api_origin3_stats():
    total = 0
    labeled = 0
    if os.path.exists(ORIGIN3_IMAGES_DIR):
        for f in os.listdir(ORIGIN3_IMAGES_DIR):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                total += 1
                stem = os.path.splitext(f)[0]
                lbl_path = os.path.join(ORIGIN3_LABELS_DIR, f"{stem}.txt")
                if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                    labeled += 1
    unlabeled = max(0, total - labeled)
    import torch
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU (Optimized)"
    return jsonify({
        "total": total,
        "labeled": labeled,
        "unlabeled": unlabeled,
        "device": device
    })

@app.route("/api/origin3/train/start", methods=["POST"])
def api_origin3_train_start():
    data = request.get_json(silent=True) or {}
    epochs = int(data.get("epochs", 30))
    batch = int(data.get("batch_size", 8))
    weights = data.get("weights_choice", "best")

    success, msg = origin3_training_mgr.start(epochs=epochs, batch=batch, weights_choice=weights)
    if success:
        return jsonify({"success": True, "message": msg})
    return jsonify({"success": False, "error": msg}), 400

@app.route("/api/origin3/train/status", methods=["GET"])
def api_origin3_train_status():
    return jsonify({
        "status": origin3_training_mgr.status,
        "current_epoch": origin3_training_mgr.current_epoch,
        "total_epochs": origin3_training_mgr.total_epochs,
        "progress_percent": origin3_training_mgr.progress_percent,
        "logs": origin3_training_mgr.logs[-50:],
        "error": origin3_training_mgr.error_msg
    })

def open_browser(port):
    time.sleep(1.2)
    webbrowser.open(f"http://localhost:{port}")

def main():
    parser = argparse.ArgumentParser(description="Wire Harness Pose Inspection Web Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run web server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    get_model()

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    print(f"\n=======================================================")
    print(f" WIRE HARNESS INSPECTION & ANNOTATION STUDIO")
    print(f" Web Dashboard URL: http://localhost:{args.port}")
    print(f"=======================================================\n")

    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
