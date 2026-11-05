"""
Solder Ball Inspection & Active Retraining Platform
Unified Local Web Application (Flask + OpenCV + Ultralytics YOLO)

Features:
1. Solder Ball Detection & Counting:
   - Upload PCB image (JPG, PNG)
   - Real-time Camera capture (webcam / microscope camera)
   - Select from existing dataset boards
   - Live bounding boxes, confidence tags, and total count readout (PASS / DEFECT)
   - Interactive zoomable inspection canvas
   - Export annotated images and CSV reports

2. Active Retraining Studio:
   - Multi-file upload for 10-15 new PCB images
   - High-precision in-browser annotation (Wheel zoom, Quick-Click marker, Box drag)
   - AI Pre-Annotation assist (uses current model to pre-label)
   - One-click fine-tuning/retraining in background thread with live progress & logs
   - Automatic model hot-reloading upon completion
"""

import os
import sys
import glob
import json
import time
import base64
import shutil
import threading
import subprocess
import webbrowser
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_file
from ultralytics import YOLO

app = Flask(__name__)

# WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
# DATASET_DIR = os.path.join(WORKSPACE_DIR, 'dataset')
# ANNOTATIONS_DIR = os.path.join(WORKSPACE_DIR, 'annotations')
# MODELS_DIR = os.path.join(WORKSPACE_DIR, 'models')
# RETRAIN_IMG_DIR = os.path.join(WORKSPACE_DIR, 'retrain_data', 'images')
# RETRAIN_LBL_DIR = os.path.join(WORKSPACE_DIR, 'retrain_data', 'labels')
# DATA_YAML_PATH = os.path.join(WORKSPACE_DIR, 'data.yaml')

# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(RETRAIN_IMG_DIR, exist_ok=True)
# os.makedirs(RETRAIN_LBL_DIR, exist_ok=True)

# MODEL_PATH = os.path.join(MODELS_DIR, 'best_solderball_detector.pt')
# CURRENT_MODEL = None
# MODEL_LOCK = threading.Lock()

# # ----------------- SOLDER PROJECT MANAGEMENT -----------------
# SOLDER_PROJECTS_DIR = os.path.join(WORKSPACE_DIR, 'projects_solder')
# ACTIVE_SOLDER_PROJECT_ID = 'default'
# os.makedirs(SOLDER_PROJECTS_DIR, exist_ok=True)

# ----------------- APPLICATION RESOURCE & DATA PATHS -----------------

# Bundled application resources (read-only when packaged)
WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset")
BUNDLED_ANNOTATIONS_DIR = os.path.join(WORKSPACE_DIR, "annotations")
BUNDLED_MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")

BUNDLED_RETRAIN_IMG_DIR = os.path.join(
    WORKSPACE_DIR, "retrain_data", "images"
)
BUNDLED_RETRAIN_LBL_DIR = os.path.join(
    WORKSPACE_DIR, "retrain_data", "labels"
)

# Writable application data directory
if getattr(sys, "frozen", False):
    DATA_DIR = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "QualityInspection",
        "PCB",
        "solderball_detection_platform"
    )
else:
    # Preserve the existing development environment
    DATA_DIR = WORKSPACE_DIR

DATASET_DIR = os.path.join(DATA_DIR, "dataset")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
MODELS_DIR = os.path.join(DATA_DIR, "models")
RETRAIN_IMG_DIR = os.path.join(DATA_DIR, "retrain_data", "images")
RETRAIN_LBL_DIR = os.path.join(DATA_DIR, "retrain_data", "labels")
DATA_YAML_PATH = os.path.join(DATA_DIR, "data.yaml")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RETRAIN_IMG_DIR, exist_ok=True)
os.makedirs(RETRAIN_LBL_DIR, exist_ok=True)

def copy_missing_files(source_dir, destination_dir):
    if not os.path.isdir(source_dir):
        return

    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)

        if relative_path == ".":
            target_dir = destination_dir
        else:
            target_dir = os.path.join(destination_dir, relative_path)

        os.makedirs(target_dir, exist_ok=True)

        for filename in files:
            source_file = os.path.join(root, filename)
            destination_file = os.path.join(target_dir, filename)

            # Never overwrite existing user data
            if not os.path.exists(destination_file):
                shutil.copy2(source_file, destination_file)


# Copy bundled seed data only when running as a packaged application
if getattr(sys, "frozen", False):
    copy_missing_files(SAMPLE_DATASET_DIR, DATASET_DIR)
    copy_missing_files(BUNDLED_ANNOTATIONS_DIR, ANNOTATIONS_DIR)
    copy_missing_files(BUNDLED_MODELS_DIR, MODELS_DIR)
    copy_missing_files(BUNDLED_RETRAIN_IMG_DIR, RETRAIN_IMG_DIR)
    copy_missing_files(BUNDLED_RETRAIN_LBL_DIR, RETRAIN_LBL_DIR)

    
MODEL_PATH = os.path.join(MODELS_DIR, "best_solderball_detector.pt")
CURRENT_MODEL = None
MODEL_LOCK = threading.Lock()

# ----------------- SOLDER PROJECT MANAGEMENT -----------------
SOLDER_PROJECTS_DIR = os.path.join(DATA_DIR, "projects_solder")
ACTIVE_SOLDER_PROJECT_ID = "default"
os.makedirs(SOLDER_PROJECTS_DIR, exist_ok=True)


def get_solder_paths(project_id=None):
    pid = project_id or ACTIVE_SOLDER_PROJECT_ID
    if pid == 'default':
        return {
            'id': 'default',
            'is_default': True,
            'proj_dir': DATA_DIR,
            'dataset_dir': DATASET_DIR,
            'annotations_dir': ANNOTATIONS_DIR,
            'retrain_img_dir': RETRAIN_IMG_DIR,
            'retrain_lbl_dir': RETRAIN_LBL_DIR,
            'data_dir': os.path.join(DATA_DIR, 'data'),
            'data_yaml_path': DATA_YAML_PATH,
            'model_path': MODEL_PATH,
            'runs_dir': os.path.join(DATA_DIR, 'runs')
        }
    proj_dir = os.path.join(SOLDER_PROJECTS_DIR, pid)
    return {
        'id': pid,
        'is_default': False,
        'proj_dir': proj_dir,
        'dataset_dir': os.path.join(proj_dir, 'dataset'),
        'annotations_dir': os.path.join(proj_dir, 'annotations'),
        'retrain_img_dir': os.path.join(proj_dir, 'retrain_data', 'images'),
        'retrain_lbl_dir': os.path.join(proj_dir, 'retrain_data', 'labels'),
        'data_dir': os.path.join(proj_dir, 'data'),
        'data_yaml_path': os.path.join(proj_dir, 'data.yaml'),
        'model_path': os.path.join(proj_dir, 'models', 'best_solderball_detector.pt'),
        'runs_dir': os.path.join(proj_dir, 'runs')
    }


def list_solder_projects():
    projects = []
    def_paths = get_solder_paths('default')
    def_imgs = glob.glob(os.path.join(def_paths['retrain_img_dir'], '*.*'))
    projects.append({
        'id': 'default',
        'name': 'Default Solder Dataset',
        'is_default': True,
        'created_at': 'Pre-loaded',
        'image_count': len(def_imgs),
        'has_model': os.path.exists(def_paths['model_path'])
    })
    if os.path.exists(SOLDER_PROJECTS_DIR):
        for p_id in sorted(os.listdir(SOLDER_PROJECTS_DIR)):
            p_dir = os.path.join(SOLDER_PROJECTS_DIR, p_id)
            if os.path.isdir(p_dir):
                meta_file = os.path.join(p_dir, 'project.json')
                meta = {}
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as mf:
                            meta = json.load(mf)
                    except Exception:
                        pass
                p_paths = get_solder_paths(p_id)
                p_imgs = glob.glob(os.path.join(p_paths['retrain_img_dir'], '*.*'))
                projects.append({
                    'id': p_id,
                    'name': meta.get('name', p_id),
                    'is_default': False,
                    'created_at': meta.get('created_at', 'Unknown'),
                    'image_count': len(p_imgs),
                    'has_model': os.path.exists(p_paths['model_path'])
                })
    return projects


def create_solder_project(name):
    global ACTIVE_SOLDER_PROJECT_ID
    import re
    clean_name = re.sub(r'[^a-zA-Z0-9_\-\s]', '', name.strip()) or 'Solder_Project'
    pid = 'solder_' + time.strftime('%Y%m%d_%H%M%S')
    paths = get_solder_paths(pid)
    for p in [
        paths['dataset_dir'],
        paths['annotations_dir'],
        paths['retrain_img_dir'],
        paths['retrain_lbl_dir'],
        paths['data_dir'],
        os.path.join(paths['proj_dir'], 'models'),
        paths['runs_dir']
    ]:
        os.makedirs(p, exist_ok=True)

    norm_data_dir = paths['data_dir'].replace('\\', '/')
    yaml_content = f"# Solder Ball Dataset\npath: {norm_data_dir}\ntrain: images/train\nval: images/val\n\nnames:\n  0: solder_ball\n"
    with open(paths['data_yaml_path'], 'w', encoding='utf-8') as yf:
        yf.write(yaml_content)

    meta = {
        'id': pid,
        'name': clean_name,
        'type': 'solder_detection',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(paths['proj_dir'], 'project.json'), 'w', encoding='utf-8') as mf:
        json.dump(meta, mf, indent=2)

    ACTIVE_SOLDER_PROJECT_ID = pid
    return meta


def delete_solder_project(pid):
    global ACTIVE_SOLDER_PROJECT_ID
    if pid == 'default':
        return False, 'Cannot delete default project.'
    p_dir = os.path.join(SOLDER_PROJECTS_DIR, pid)
    if os.path.exists(p_dir):
        shutil.rmtree(p_dir, ignore_errors=True)
    if ACTIVE_SOLDER_PROJECT_ID == pid:
        ACTIVE_SOLDER_PROJECT_ID = 'default'
    return True, 'Project deleted.'


def split_solder_dataset_internal(paths, val_ratio=0.2, seed=42):
    import random
    random.seed(seed)
    dataset_dir = paths['dataset_dir']
    ann_dir = paths['annotations_dir']
    data_dir = paths['data_dir']

    train_img_dir = os.path.join(data_dir, 'images', 'train')
    val_img_dir = os.path.join(data_dir, 'images', 'val')
    train_lbl_dir = os.path.join(data_dir, 'labels', 'train')
    val_lbl_dir = os.path.join(data_dir, 'labels', 'val')

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, '*')):
            if os.path.isfile(f):
                try: os.remove(f)
                except Exception: pass

    img_files = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')) +
                       glob.glob(os.path.join(dataset_dir, '*.png')) +
                       glob.glob(os.path.join(dataset_dir, '*.jpeg')))
    pairs = []
    for img_p in img_files:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(ann_dir, f"{stem}.txt")
        if os.path.exists(lbl_p) and os.path.getsize(lbl_p) > 0:
            pairs.append((img_p, lbl_p))

    if not pairs:
        return False, "No labeled images found in project dataset. Please annotate at least one image before retraining."

    random.shuffle(pairs)
    val_cnt = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 1
    val_pairs = pairs[:val_cnt]
    train_pairs = pairs[val_cnt:] if len(pairs) > 1 else pairs

    for img_p, lbl_p in train_pairs:
        shutil.copy2(img_p, train_img_dir)
        shutil.copy2(lbl_p, train_lbl_dir)

    for img_p, lbl_p in val_pairs:
        shutil.copy2(img_p, val_img_dir)
        shutil.copy2(lbl_p, val_lbl_dir)

    norm_data_dir = data_dir.replace('\\', '/')
    yaml_content = f"# Solder Ball Dataset\npath: {norm_data_dir}\ntrain: images/train\nval: images/val\n\nnames:\n  0: solder_ball\n"
    with open(paths['data_yaml_path'], 'w', encoding='utf-8') as yf:
        yf.write(yaml_content)

    return True, f"Dataset split complete: {len(train_pairs)} train, {len(val_pairs)} val images."

def get_or_load_model(project_id=None):
    global CURRENT_MODEL
    paths = get_solder_paths(project_id)
    model_to_load = paths['model_path'] if os.path.exists(paths['model_path']) else MODEL_PATH
    with MODEL_LOCK:
        if CURRENT_MODEL is None or getattr(CURRENT_MODEL, '_loaded_path', None) != model_to_load:
            if os.path.exists(model_to_load):
                try:
                    CURRENT_MODEL = YOLO(model_to_load)
                    CURRENT_MODEL._loaded_path = model_to_load
                except Exception as e:
                    print(f"Error loading model {model_to_load}: {e}")
                    CURRENT_MODEL = None
            elif os.path.exists(MODEL_PATH):
                try:
                    CURRENT_MODEL = YOLO(MODEL_PATH)
                    CURRENT_MODEL._loaded_path = MODEL_PATH
                except Exception as e:
                    print(f"Error loading fallback model: {e}")
                    CURRENT_MODEL = None
        return CURRENT_MODEL

# Global retraining state
training_state = {
    'running': False,
    'completed': False,
    'error': None,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'loss': 0.0,
    'map50': 0.0,
    'logs': [],
    'model_version': time.strftime("%Y-%m-%d %H:%M")
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solder Ball Inspection & Retraining Platform</title>
    <style>
        :root {
            --bg-dark: #121316;
            --bg-card: #1c1d23;
            --bg-card-hover: #24252e;
            --border-color: #2e303d;
            --primary: #0088ff;
            --primary-hover: #0070d6;
            --success: #00e676;
            --danger: #ff3b30;
            --warning: #ffb300;
            --text-main: #f0f2f5;
            --text-muted: #9aa0a6;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-dark);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Top Navigation Header */
        header {
            height: 60px;
            background: #181920;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            flex-shrink: 0;
            z-index: 100;
        }
        .logo-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #0088ff, #00e676);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: 800;
            color: #fff;
        }
        .logo-title {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.3px;
        }
        .logo-sub {
            font-size: 12px;
            color: var(--text-muted);
            margin-left: 8px;
            border-left: 1px solid var(--border-color);
            padding-left: 8px;
        }

        .nav-tabs {
            display: flex;
            gap: 8px;
            background: #101115;
            padding: 4px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 18px;
            font-size: 13px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .tab-btn:hover { color: #fff; }
        .tab-btn.active {
            background: var(--primary);
            color: #fff;
            box-shadow: 0 2px 8px rgba(0, 136, 255, 0.35);
        }

        .header-status {
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 8px var(--success);
        }

        /* Main Workspace Container */
        main {
            flex: 1;
            overflow: hidden;
            position: relative;
        }
        .view-section {
            width: 100%;
            height: 100%;
            display: none;
            overflow: hidden;
        }
        .view-section.active {
            display: flex;
        }

        /* Button & Form Controls */
        .btn {
            background: #2a2c36;
            color: #fff;
            border: 1px solid var(--border-color);
            padding: 9px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.15s;
        }
        .btn:hover { background: #353845; }
        .btn-primary {
            background: var(--primary);
            border-color: #1a96ff;
        }
        .btn-primary:hover { background: var(--primary-hover); }
        .btn-success {
            background: #117a43;
            border-color: var(--success);
        }
        .btn-success:hover { background: #159452; }
        .btn-danger {
            background: #7a1c1c;
            border-color: var(--danger);
            color: #ffb3b3;
        }
        .btn-danger:hover { background: #962222; color: #fff; }

        /* VIEW 1: INSPECTION LAYOUT */
        .inspect-sidebar {
            width: 360px;
            background: var(--bg-card);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 20px;
            overflow-y: auto;
            flex-shrink: 0;
        }
        .sidebar-card {
            background: #15161c;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
        }
        .sidebar-card-title {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Source selector pills */
        .source-pills {
            display: flex;
            gap: 6px;
            margin-bottom: 12px;
        }
        .source-pill {
            flex: 1;
            padding: 8px 0;
            text-align: center;
            font-size: 12px;
            font-weight: 600;
            background: #20222a;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            color: var(--text-muted);
        }
        .source-pill.active {
            background: var(--primary);
            border-color: #2b98ff;
            color: #fff;
        }

        /* Upload drop zone */
        .upload-drop-zone {
            border: 2px dashed #3a3d4d;
            border-radius: 8px;
            padding: 24px 16px;
            text-align: center;
            cursor: pointer;
            background: #14151a;
            transition: all 0.2s;
        }
        .upload-drop-zone:hover {
            border-color: var(--primary);
            background: #171a24;
        }

        /* Camera live box */
        #camera-container {
            display: none;
            margin-top: 10px;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            border: 1px solid var(--border-color);
            position: relative;
        }
        #camera-video {
            width: 100%;
            height: auto;
            display: block;
        }
        .camera-actions {
            padding: 8px;
            display: flex;
            gap: 8px;
            background: #15161c;
        }

        /* Slider styles */
        .slider-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-bottom: 12px;
        }
        .slider-label-row {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
        }
        .slider-input {
            width: 100%;
            accent-color: var(--primary);
            cursor: pointer;
        }

        /* Results Display */
        .inspect-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0f1013;
            position: relative;
            overflow: hidden;
        }

        /* Stat banner */
        .stats-banner {
            height: 70px;
            background: #181920;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            padding: 0 24px;
            gap: 30px;
            flex-shrink: 0;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
        }
        .stat-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }
        .stat-value {
            font-size: 24px;
            font-weight: 800;
            color: #fff;
        }
        .badge-status {
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .badge-pass { background: rgba(0, 230, 118, 0.15); color: var(--success); border: 1px solid var(--success); }
        .badge-defect { background: rgba(255, 59, 48, 0.15); color: var(--danger); border: 1px solid var(--danger); }

        /* Canvas viewport */
        .canvas-viewport {
            flex: 1;
            position: relative;
            background: #0c0d10;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #inspect-canvas {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
            border-radius: 4px;
        }
        .canvas-controls-overlay {
            position: absolute;
            bottom: 16px;
            right: 16px;
            background: rgba(24, 25, 32, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 6px 10px;
            display: flex;
            gap: 8px;
            z-index: 10;
        }

        /* VIEW 2: RETRAINING STUDIO LAYOUT */
        .retrain-sidebar {
            width: 380px;
            background: var(--bg-card);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 16px;
            overflow-y: auto;
            flex-shrink: 0;
        }
        .retrain-image-list {
            flex: 1;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: #14151a;
            padding: 6px;
            max-height: 280px;
        }
        .retrain-item {
            padding: 8px 12px;
            border-radius: 6px;
            background: #1e1f26;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            font-size: 13px;
            border: 1px solid transparent;
            transition: all 0.15s;
        }
        .retrain-item:hover { background: #282a33; }
        .retrain-item.active {
            background: #17385c;
            border-color: var(--primary);
            color: #fff;
            font-weight: 600;
        }
        .tag-pill {
            font-size: 11px;
            padding: 2px 7px;
            border-radius: 10px;
            background: #333644;
            color: #aaa;
        }
        .tag-pill.done {
            background: #0f4a27;
            color: #00ff77;
            font-weight: 700;
        }

        /* Retrain Workspace */
        .retrain-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0c0d10;
            position: relative;
        }
        .retrain-toolbar {
            height: 52px;
            background: #1a1b22;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            gap: 12px;
        }

        /* Interactive Canvas for Annotation */
        #retrain-viewport {
            flex: 1;
            position: relative;
            background: #0c0d10;
            overflow: hidden;
            cursor: crosshair;
        }
        #retrain-canvas {
            position: absolute;
            top: 0;
            left: 0;
            transform-origin: 0 0;
        }

        /* Progress Card */
        .progress-box {
            background: #15161c;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
        }
        .progress-track {
            height: 8px;
            background: #2a2c36;
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        }
        .progress-bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--primary), var(--success));
            transition: width 0.3s;
        }

        /* Terminal Log Output */
        .console-box {
            background: #090a0c;
            border: 1px solid #22242d;
            border-radius: 6px;
            padding: 10px;
            font-family: Consolas, monospace;
            font-size: 11px;
            color: #98c379;
            height: 120px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        /* Loading Spinner */
        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header>
        <div class="logo-group">
            <div class="logo-icon">AOI</div>
            <div class="logo-title">Solder Ball Platform</div>
            <div class="logo-sub">Automated PCB Quality Inspection & Continuous Learning</div>
        </div>

        <div class="nav-tabs">
            <button class="tab-btn active" onclick="switchView('inspect')">
                <span>🔍 Inspect & Count</span>
            </button>
            <button class="tab-btn" onclick="switchView('retrain')">
                <span>🔄 Retrain Model</span>
            </button>
        </div>

        <div class="header-status">
            <div class="status-dot"></div>
            <span id="model-status-text">YOLOv8 Active (86.2% Precision)</span>
        </div>
    </header>

    <!-- Main Workspace -->
    <main>

        <!-- =================== VIEW 1: INSPECT & COUNT =================== -->
        <section id="view-inspect" class="view-section active">
            <!-- Sidebar -->
            <div class="inspect-sidebar">
                <div class="sidebar-card">
                    <div class="sidebar-card-title">1. Select PCB Input Source</div>
                    
                    <div class="source-pills">
                        <div class="source-pill active" id="pill-upload" onclick="setInspectSource('upload')">📁 Upload</div>
                        <div class="source-pill" id="pill-camera" onclick="setInspectSource('camera')">📸 Camera</div>
                        <div class="source-pill" id="pill-dataset" onclick="setInspectSource('dataset')">🖼️ Dataset</div>
                    </div>

                    <!-- Upload Form -->
                    <div id="source-upload-box">
                        <div class="upload-drop-zone" onclick="document.getElementById('inspect-file-input').click()">
                            <div style="font-size: 28px; margin-bottom: 6px;">📂</div>
                            <div style="font-size: 13px; font-weight: 600;">Click to Browse or Drag Image</div>
                            <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Supports JPG, PNG (up to 4K)</div>
                        </div>
                        <input type="file" id="inspect-file-input" accept="image/*" style="display: none;" onchange="handleInspectFileUpload(this)">
                    </div>

                    <!-- Camera Snap Form -->
                    <div id="source-camera-box" style="display: none;">
                        <button class="btn btn-primary" style="width: 100%;" onclick="startCamera()">📷 Open Live Camera</button>
                        <div id="camera-container">
                            <video id="camera-video" autoplay playsinline></video>
                            <div class="camera-actions">
                                <button class="btn btn-success" style="flex: 1;" onclick="captureCameraPhoto()">📸 Snap & Detect</button>
                                <button class="btn btn-danger" onclick="stopCamera()">✕ Close</button>
                            </div>
                        </div>
                    </div>

                    <!-- Dataset Selector Form -->
                    <div id="source-dataset-box" style="display: none;">
                        <select id="dataset-image-select" class="btn" style="width: 100%; text-align: left;" onchange="loadDatasetImage(this.value)">
                            <option value="">Select a board from dataset...</option>
                        </select>
                    </div>
                </div>

                <!-- Detection Settings -->
                <div class="sidebar-card">
                    <div class="sidebar-card-title">2. Detection Sensitivity</div>

                    <div class="slider-group">
                        <div class="slider-label-row">
                            <span>Confidence Threshold:</span>
                            <span id="conf-val" style="color: var(--primary); font-weight: 700;">0.25</span>
                        </div>
                        <input type="range" class="slider-input" id="conf-slider" min="0.05" max="0.95" step="0.05" value="0.25" oninput="updateConfidence(this.value)">
                    </div>

                    <div class="slider-group">
                        <div class="slider-label-row">
                            <span>IoU NMS Threshold:</span>
                            <span id="iou-val" style="color: var(--primary); font-weight: 700;">0.45</span>
                        </div>
                        <input type="range" class="slider-input" id="iou-slider" min="0.10" max="0.90" step="0.05" value="0.45" oninput="updateIoU(this.value)">
                    </div>

                    <button class="btn btn-primary" style="width: 100%; margin-top: 8px;" onclick="reRunDetection()">
                        ⚡ Run Detection
                    </button>
                </div>

                <!-- Export & Reports -->
                <div class="sidebar-card">
                    <div class="sidebar-card-title">3. Export Inspection Log</div>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn" style="flex: 1;" onclick="downloadAnnotatedImage()">📥 Image</button>
                        <button class="btn" style="flex: 1;" onclick="downloadCSVReport()">📊 CSV Log</button>
                    </div>
                </div>
            </div>

            <!-- Inspect Main Content -->
            <div class="inspect-main">
                <!-- Header Stats Banner -->
                <div class="stats-banner">
                    <div class="stat-item">
                        <span class="stat-label">Solder Balls Found</span>
                        <span id="banner-count" class="stat-value" style="color: var(--success); font-size: 26px;">-</span>
                    </div>

                    <div class="stat-item">
                        <span class="stat-label">Inference Time</span>
                        <span id="banner-speed" class="stat-value" style="font-size: 18px; color: var(--text-muted);">-</span>
                    </div>

                    <div class="stat-item" style="margin-left: 15px;">
                        <button id="btn-open-correction" class="btn" onclick="openCorrectionModal()" style="display: none; background: #8f5800; border-color: #f59e0b; color: #fff; font-weight: 700; padding: 7px 14px; font-size: 12px;">
                            ✏️ Wrong Detection? Correct It
                        </button>
                    </div>

                    <div class="stat-item" style="margin-left: auto;">
                        <span class="stat-label">View Mode</span>
                        <div style="display: flex; gap: 6px; margin-top: 4px;">
                            <button id="btn-toggle-annotated" class="btn active" style="padding: 5px 10px; font-size: 12px;" onclick="toggleViewMode('annotated')">Detections</button>
                            <button id="btn-toggle-original" class="btn" style="padding: 5px 10px; font-size: 12px;" onclick="toggleViewMode('original')">Original</button>
                        </div>
                    </div>
                </div>

                <!-- Canvas Visualizer -->
                <div class="canvas-viewport">
                    <img id="inspect-canvas" alt="PCB Image">
                    <div id="inspect-loading" style="display: none; position: absolute; text-align: center;">
                        <div class="spinner" style="width: 40px; height: 40px; border-width: 4px;"></div>
                        <div style="margin-top: 10px; font-size: 14px; font-weight: 600;">Analyzing PCB with YOLOv8...</div>
                    </div>
                </div>
            </div>
        </section>


        <!-- =================== VIEW 2: RETRAINING STUDIO =================== -->
        <section id="view-retrain" class="view-section">
            <!-- Retrain Sidebar -->
            <div class="retrain-sidebar">
                <!-- Solder Project Workspace Card -->
                <div class="sidebar-card">
                    <div class="sidebar-card-title">
                        <span>Solder Inspection Project</span>
                        <button class="btn btn-primary" style="padding: 2px 8px; font-size: 11px;" onclick="openNewSolderProjectModal()">+ New Project</button>
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center; margin-top: 6px;">
                        <select id="solderProjectSelect" style="flex: 1; min-width: 0; padding: 6px 8px; font-size: 12px; background: #121316; color: #f0f2f5; border: 1px solid #2e303d; border-radius: 4px; text-overflow: ellipsis; white-space: nowrap; overflow: hidden;" onchange="switchSolderProject(this.value)">
                            <!-- Populated dynamically -->
                        </select>
                        <button id="btnDeleteSolderProject" class="btn btn-danger" style="flex-shrink: 0; padding: 6px 10px; font-size: 12px;" title="Delete Project" onclick="deleteSolderProject()">🗑️</button>
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
                        <span class="tag-pill" style="background: rgba(0, 136, 255, 0.15); color: #0088ff; border: 1px solid rgba(0, 136, 255, 0.3); white-space: nowrap;">
                            YOLOv8 &bull; Solder
                        </span>
                        <span id="solderProjectBadge" style="color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Default Dataset</span>
                    </div>
                </div>

                <div class="sidebar-card">
                    <div class="sidebar-card-title">1. Upload Staging Images (10-15)</div>
                    <div class="upload-drop-zone" onclick="document.getElementById('retrain-file-input').click()">
                        <div style="font-size: 26px; margin-bottom: 6px;">📤</div>
                        <div style="font-size: 13px; font-weight: 600;">Upload 10–15 Images</div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Select multiple JPG/PNG files</div>
                    </div>
                    <input type="file" id="retrain-file-input" multiple accept="image/*" style="display: none;" onchange="handleRetrainUpload(this)">
                </div>

                <!-- Staged Images List -->
                <div class="sidebar-card" style="flex: 1; display: flex; flex-direction: column;">
                    <div class="sidebar-card-title">
                        <span>Staged Images</span>
                        <span id="retrain-progress-badge" class="tag-pill">0 / 0</span>
                    </div>
                    <div id="retrain-images-list" class="retrain-image-list"></div>
                </div>

                <!-- Retrain Trigger Box -->
                <div class="sidebar-card">
                    <div class="sidebar-card-title">2. Training Parameters</div>
                    <div class="slider-group">
                        <div class="slider-label-row">
                            <span>Epochs:</span>
                            <span id="retrain-epochs-val" style="color: var(--primary); font-weight: 700;">20</span>
                        </div>
                        <input type="range" class="slider-input" id="retrain-epochs-slider" min="5" max="50" step="5" value="20" oninput="document.getElementById('retrain-epochs-val').innerText = this.value">
                    </div>

                    <div style="margin: 8px 0 10px 0;">
                        <label style="font-size: 11px; color: var(--text-muted); display: block; margin-bottom: 4px;">Base Weights:</label>
                        <select id="retrain-weights-select" style="width: 100%; padding: 6px 8px; font-size: 12px; background: #121316; color: #f0f2f5; border: 1px solid #2e303d; border-radius: 4px;">
                            <option value="nano">Train Fresh from Scratch (yolov8n.pt)</option>
                            <option value="best">Fine-tune from Project Checkpoint (best.pt)</option>
                        </select>
                    </div>

                    <button id="btn-start-retrain" class="btn btn-success" style="width: 100%; font-size: 14px; padding: 12px;" onclick="startRetraining()">
                        🚀 Start Retraining Model
                    </button>
                </div>

                <!-- Live Training Status -->
                <div class="progress-box" id="train-status-box" style="display: none;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 700;">
                        <span id="train-status-title">Training in Progress...</span>
                        <span id="train-percent">0%</span>
                    </div>
                    <div class="progress-track">
                        <div id="train-progress-fill" class="progress-bar-fill"></div>
                    </div>
                    <div class="console-box" id="train-console"></div>
                </div>
            </div>

            <!-- Retrain Main Canvas Area -->
            <div class="retrain-main">
                <!-- Toolbar -->
                <div class="retrain-toolbar">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <button class="btn" onclick="prevRetrainImage()">◀ Prev (A)</button>
                        <button class="btn" onclick="nextRetrainImage()">Next (D) ▶</button>
                        <span id="retrain-filename" style="font-size: 13px; font-weight: 600; margin-left: 8px;"></span>
                    </div>

                    <div style="display: flex; align-items: center; gap: 8px;">
                        <button id="btn-mode-drag" class="btn active" onclick="setRetrainMode('drag')">🔲 Box Drag (1)</button>
                        <button id="btn-mode-click" class="btn" onclick="setRetrainMode('click')">🎯 Quick Click (2)</button>
                    </div>

                    <div style="display: flex; align-items: center; gap: 8px;">
                        <button class="btn btn-primary" onclick="runAIPreAnnotate()" title="Predicts boxes using current model">
                            ✨ AI Pre-Annotate
                        </button>
                        <button class="btn" onclick="undoRetrainBox()">↩ Undo (Z)</button>
                        <button class="btn btn-danger" onclick="deleteCurrentRetrainImage()">🗑 Delete Image</button>
                        <button class="btn btn-success" onclick="saveRetrainAnnotations(true)">💾 Save</button>
                    </div>
                </div>

                <!-- Canvas -->
                <div id="retrain-viewport">
                    <canvas id="retrain-canvas"></canvas>
                </div>
            </div>
        </section>

        <!-- ================= FULLSCREEN ERROR CORRECTION MODAL ================= -->
        <div id="correction-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: #0c0d10; z-index: 3000; flex-direction: column;">
            <!-- Top Toolbar -->
            <div style="height: 56px; background: #181920; border-bottom: 1px solid var(--border-color); display: flex; align-items: center; justify-content: space-between; padding: 0 20px; gap: 12px; flex-shrink: 0;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 16px; font-weight: 700; color: #fff;">✏️ Correct Solder Ball Detections</span>
                    <span id="corr-filename" style="font-size: 12px; color: var(--text-muted); background: #22242d; padding: 4px 10px; border-radius: 4px;"></span>
                    <span id="corr-count-badge" style="font-size: 12px; font-weight: 700; color: #00ff88; background: #0f4a27; padding: 4px 10px; border-radius: 12px;">0 Balls</span>
                </div>

                <div style="display: flex; align-items: center; gap: 8px;">
                    <button id="btn-corr-click" class="btn active" onclick="setCorrMode('click')" title="Click directly on solder balls to add">🎯 Quick Click (2)</button>
                    <button id="btn-corr-drag" class="btn" onclick="setCorrMode('drag')" title="Drag custom rectangles">🔲 Box Drag (1)</button>
                    <button class="btn" onclick="undoCorrBox()" title="Shortcut: Ctrl+Z">↩ Undo (Z)</button>
                    <button class="btn btn-danger" onclick="clearCorrBoxes()">🗑 Clear All</button>
                </div>

                <div style="display: flex; align-items: center; gap: 10px;">
                    <button class="btn" onclick="closeCorrectionModal()">Cancel</button>
                    <button class="btn btn-primary" onclick="submitCorrection('queue')" title="Saves this correction into retraining staging">
                        💾 Save to Retrain Queue
                    </button>
                    <button class="btn btn-success" onclick="submitCorrection('retrain_now')" title="Saves and starts retraining immediately">
                        🚀 Save & Retrain Now
                    </button>
                </div>
            </div>

            <!-- Canvas Viewport -->
            <div id="corr-viewport" style="flex: 1; position: relative; background: #08090b; overflow: hidden; cursor: crosshair;">
                <canvas id="corr-canvas" style="position: absolute; top: 0; left: 0; transform-origin: 0 0;"></canvas>
            </div>

            <!-- Help Bar -->
            <div style="height: 34px; background: #13141a; border-top: 1px solid var(--border-color); display: flex; align-items: center; justify-content: space-between; padding: 0 20px; font-size: 12px; color: #888;">
                <div>
                    💡 <b>How to correct:</b> Click anywhere to ADD a missing ball &bull; Click on a box (highlighted in red) to REMOVE it &bull; Wheel to Zoom &bull; Right-Click or Space to Pan
                </div>
                <div>
                    Zoom: <span id="corr-zoom-text" style="color: #00ff88; font-weight: bold;">100%</span>
                </div>
            </div>
        </div>

    </main>

    <!-- Modal: New Solderball Project -->
    <div id="newSolderProjectModal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.75); z-index: 9999; align-items: center; justify-content: center; backdrop-filter: blur(4px);">
        <div style="background: #1c1d23; border: 1px solid #2e303d; border-radius: 8px; width: 440px; max-width: 90vw; padding: 22px; box-shadow: 0 16px 40px rgba(0,0,0,0.6);">
            <div style="font-size: 16px; font-weight: 700; color: #f0f2f5; margin-bottom: 6px;">Create New Solder Project</div>
            <div style="font-size: 12px; color: var(--text-muted); line-height: 1.4; margin-bottom: 16px;">
                Start with a clean, empty workspace for Solder Ball Inspection. Upload new PCB images, annotate solder balls with clicks or boxes, and retrain a new YOLOv8 detector from scratch.
            </div>
            <label style="font-size: 12px; font-weight: 600; color: #f0f2f5; display: block; margin-bottom: 6px;">Project Name:</label>
            <input type="text" id="newSolderProjectNameInput" placeholder="e.g. Solder Line 2 Batch C" style="width: 100%; box-sizing: border-box; padding: 8px 12px; font-size: 13px; background: #121316; color: #f0f2f5; border: 1px solid #2e303d; border-radius: 4px; margin-bottom: 16px;" onkeydown="if(event.key==='Enter') submitCreateSolderProject()">
            <div style="display: flex; justify-content: flex-end; gap: 8px;">
                <button class="btn" onclick="closeNewSolderProjectModal()">Cancel</button>
                <button class="btn btn-primary" onclick="submitCreateSolderProject()">Create & Open Project</button>
            </div>
        </div>
    </div>

    <!-- Hidden canvas for camera snapshot -->
    <canvas id="hidden-snap-canvas" style="display: none;"></canvas>

    <script>
        // ================= GLOBAL NAVIGATION =================
        function switchView(viewName) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));

            if (viewName === 'inspect') {
                document.querySelectorAll('.tab-btn')[0].classList.add('active');
                document.getElementById('view-inspect').classList.add('active');
            } else {
                document.querySelectorAll('.tab-btn')[1].classList.add('active');
                document.getElementById('view-retrain').classList.add('active');
                fetchRetrainImages();
            }
        }

        // ================= VIEW 1: INSPECT & DETECT =================
        let currentInspectImageB64 = null;
        let currentAnnotatedB64 = null;
        let currentDetections = [];
        let currentFilename = 'pcb_inspection.jpg';
        let mediaStream = null;

        function setInspectSource(mode) {
            document.getElementById('pill-upload').classList.toggle('active', mode === 'upload');
            document.getElementById('pill-camera').classList.toggle('active', mode === 'camera');
            document.getElementById('pill-dataset').classList.toggle('active', mode === 'dataset');

            document.getElementById('source-upload-box').style.display = mode === 'upload' ? 'block' : 'none';
            document.getElementById('source-camera-box').style.display = mode === 'camera' ? 'block' : 'none';
            document.getElementById('source-dataset-box').style.display = mode === 'dataset' ? 'block' : 'none';

            if (mode !== 'camera' && mediaStream) {
                stopCamera();
            }
            if (mode === 'dataset') {
                loadDatasetList();
            }
        }

        function updateConfidence(val) {
            document.getElementById('conf-val').innerText = val;
        }
        function updateIoU(val) {
            document.getElementById('iou-val').innerText = val;
        }

        function handleInspectFileUpload(input) {
            if (input.files && input.files[0]) {
                const file = input.files[0];
                currentFilename = file.name;
                const reader = new FileReader();
                reader.onload = (e) => {
                    currentInspectImageB64 = e.target.result;
                    runDetectionOnB64(currentInspectImageB64);
                };
                reader.readAsDataURL(file);
            }
        }

        async function startCamera() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 1920 }, height: { ideal: 1080 } }
                });
                const video = document.getElementById('camera-video');
                video.srcObject = mediaStream;
                document.getElementById('camera-container').style.display = 'block';
            } catch (err) {
                alert('Could not access camera: ' + err.message);
            }
        }

        function stopCamera() {
            if (mediaStream) {
                mediaStream.getTracks().forEach(t => t.stop());
                mediaStream = null;
            }
            document.getElementById('camera-container').style.display = 'none';
        }

        function captureCameraPhoto() {
            const video = document.getElementById('camera-video');
            const hiddenCanvas = document.getElementById('hidden-snap-canvas');
            hiddenCanvas.width = video.videoWidth || 1280;
            hiddenCanvas.height = video.videoHeight || 720;
            const hctx = hiddenCanvas.getContext('2d');
            hctx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

            currentFilename = `camera_${Date.now()}.jpg`;
            currentInspectImageB64 = hiddenCanvas.toDataURL('image/jpeg', 0.95);
            runDetectionOnB64(currentInspectImageB64);
        }

        async function loadDatasetList() {
            const res = await fetch('/api/sample_images');
            const data = await res.json();
            const select = document.getElementById('dataset-image-select');
            select.innerHTML = '<option value="">Select a board from dataset...</option>';
            data.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.innerText = name;
                select.appendChild(opt);
            });
        }

        async function loadDatasetImage(imgName) {
            if (!imgName) return;
            currentFilename = imgName;
            const res = await fetch(`/api/sample_image/${encodeURIComponent(imgName)}`);
            const blob = await res.blob();
            const reader = new FileReader();
            reader.onload = (e) => {
                currentInspectImageB64 = e.target.result;
                runDetectionOnB64(currentInspectImageB64);
            };
            reader.readAsDataURL(blob);
        }

        function reRunDetection() {
            if (currentInspectImageB64) {
                runDetectionOnB64(currentInspectImageB64);
            } else {
                alert('Please upload an image, snap a photo, or pick from dataset first.');
            }
        }

        async function runDetectionOnB64(dataUrl) {
            document.getElementById('inspect-loading').style.display = 'block';
            const conf = parseFloat(document.getElementById('conf-slider').value);
            const iou = parseFloat(document.getElementById('iou-slider').value);

            try {
                const res = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_b64: dataUrl,
                        conf: conf,
                        iou: iou,
                        filename: currentFilename
                    })
                });
                const result = await res.json();

                currentDetections = result.detections;
                currentAnnotatedB64 = result.annotated_b64;

                // Update Stats Banner
                const count = result.count;
                document.getElementById('banner-count').innerText = `${count} Solder Balls Found`;
                document.getElementById('banner-speed').innerText = `${result.inference_time_ms} ms`;
                document.getElementById('btn-open-correction').style.display = 'inline-flex';

                // Display on canvas
                toggleViewMode('annotated');
            } catch (err) {
                alert('Error running detection: ' + err);
            } finally {
                document.getElementById('inspect-loading').style.display = 'none';
            }
        }

        function toggleViewMode(mode) {
            document.getElementById('btn-toggle-annotated').classList.toggle('active', mode === 'annotated');
            document.getElementById('btn-toggle-original').classList.toggle('active', mode === 'original');

            const img = document.getElementById('inspect-canvas');
            if (mode === 'annotated' && currentAnnotatedB64) {
                img.src = currentAnnotatedB64;
            } else if (currentInspectImageB64) {
                img.src = currentInspectImageB64;
            }
        }

        function downloadAnnotatedImage() {
            if (!currentAnnotatedB64) return alert('No detection result to download.');
            const a = document.createElement('a');
            a.href = currentAnnotatedB64;
            a.download = `detected_${currentFilename}`;
            a.click();
        }

        function downloadCSVReport() {
            if (!currentDetections || currentDetections.length === 0) return alert('No solder balls detected to log.');
            let csv = "Ball_Index,Confidence,X1,Y1,X2,Y2,Width_px,Height_px\\n";
            currentDetections.forEach((d, idx) => {
                const [x1, y1, x2, y2] = d.box;
                csv += `${idx + 1},${(d.conf * 100).toFixed(1)}%,${x1},${y1},${x2},${y2},${x2 - x1},${y2 - y1}\\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `report_${currentFilename.replace(/\\.[^/.]+$/, "")}.csv`;
            a.click();
        }


        // ================= VIEW 2: RETRAINING STUDIO =================
        let retrainImages = [];
        let currentRetrainIdx = 0;
        let retrainBoxes = [];
        let retrainMode = 'drag'; // 'drag' or 'click'
        let quickRadius = 24;

        let retrainImgElement = new Image();
        let rScale = 1.0, rPanX = 0, rPanY = 0;
        let rIsPanning = false, rStartPanX = 0, rStartPanY = 0;
        let rIsDrawing = false, rDrawStartX = 0, rDrawStartY = 0, rDrawCurX = 0, rDrawCurY = 0;

        const retrainViewport = document.getElementById('retrain-viewport');
        const retrainCanvas = document.getElementById('retrain-canvas');
        const rCtx = retrainCanvas.getContext('2d');

        // ================= SOLDER PROJECT MANAGEMENT =================
        let solderProjects = [];
        let activeSolderProjectId = 'default';

        async function loadSolderProjects() {
            try {
                const res = await fetch('/api/retrain/projects');
                const data = await res.json();
                activeSolderProjectId = data.active_id || 'default';
                solderProjects = data.projects || [];
                renderSolderProjectSelect();
            } catch (err) {
                console.error('Error loading solder projects:', err);
            }
        }

        function renderSolderProjectSelect() {
            const sel = document.getElementById('solderProjectSelect');
            if (!sel) return;
            sel.innerHTML = '';
            let currentProj = null;

            solderProjects.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.id;
                opt.textContent = `${p.name} (${p.image_count} img${p.has_model ? ' • ✓' : ''})`;
                if (p.id === activeSolderProjectId) {
                    opt.selected = true;
                    currentProj = p;
                }
                sel.appendChild(opt);
            });

            const badge = document.getElementById('solderProjectBadge');
            if (badge && currentProj) {
                badge.textContent = currentProj.is_default ? 'Default Dataset' : `${currentProj.image_count} imgs`;
            }
            const delBtn = document.getElementById('btnDeleteSolderProject');
            if (delBtn) {
                const isDef = !currentProj || currentProj.is_default;
                delBtn.disabled = isDef;
                delBtn.style.opacity = isDef ? '0.4' : '1.0';
                delBtn.style.cursor = isDef ? 'not-allowed' : 'pointer';
            }

            const wSel = document.getElementById('retrain-weights-select');
            if (wSel) {
                wSel.innerHTML = `
                    <option value="nano">Train Fresh from Scratch (yolov8n.pt)</option>
                    ${(currentProj && currentProj.has_model) ? '<option value="project">Fine-tune Project Model Checkpoint</option>' : ''}
                    <option value="best">Fine-tune Pre-trained Solder Model (best_solderball_detector.pt)</option>
                `;
            }
        }

        async function switchSolderProject(pid) {
            if (!pid) return;
            try {
                const res = await fetch('/api/retrain/projects/select', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ project_id: pid })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    activeSolderProjectId = data.active_id;
                    await loadSolderProjects();
                    currentRetrainIdx = 0;
                    await fetchRetrainImages();
                } else {
                    alert('Failed to switch project: ' + (data.message || 'unknown error'));
                }
            } catch (err) {
                console.error('Error switching solder project:', err);
            }
        }

        function openNewSolderProjectModal() {
            const modal = document.getElementById('newSolderProjectModal');
            if (modal) {
                modal.style.display = 'flex';
                const inp = document.getElementById('newSolderProjectNameInput');
                if (inp) {
                    inp.value = '';
                    setTimeout(() => inp.focus(), 50);
                }
            }
        }

        function closeNewSolderProjectModal() {
            const modal = document.getElementById('newSolderProjectModal');
            if (modal) modal.style.display = 'none';
        }

        async function submitCreateSolderProject() {
            const inp = document.getElementById('newSolderProjectNameInput');
            const name = inp ? inp.value.trim() : '';
            if (!name) {
                alert('Please enter a project name.');
                return;
            }
            try {
                const res = await fetch('/api/retrain/projects/create', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    closeNewSolderProjectModal();
                    activeSolderProjectId = data.project.id;
                    await loadSolderProjects();
                    currentRetrainIdx = 0;
                    await fetchRetrainImages();
                } else {
                    alert('Error creating project: ' + (data.message || 'unknown error'));
                }
            } catch (err) {
                console.error('Error creating solder project:', err);
                alert('Failed to create project: ' + err.message);
            }
        }

        async function deleteSolderProject() {
            if (activeSolderProjectId === 'default') {
                alert('Cannot delete default pre-loaded project.');
                return;
            }
            if (!confirm(`Are you sure you want to permanently delete the active project "${activeSolderProjectId}"? All images and annotations in it will be lost.`)) {
                return;
            }
            try {
                const res = await fetch('/api/retrain/projects/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ project_id: activeSolderProjectId })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    activeSolderProjectId = data.active_id || 'default';
                    await loadSolderProjects();
                    currentRetrainIdx = 0;
                    await fetchRetrainImages();
                } else {
                    alert('Failed to delete project: ' + (data.message || 'unknown error'));
                }
            } catch (err) {
                console.error('Error deleting solder project:', err);
            }
        }

        async function fetchRetrainImages() {
            try {
                const res = await fetch('/api/retrain/list');
                retrainImages = await res.json();
                renderRetrainList();
                if (retrainImages.length > 0) {
                    if (currentRetrainIdx < 0 || currentRetrainIdx >= retrainImages.length) {
                        currentRetrainIdx = 0;
                    }
                    loadRetrainImage(currentRetrainIdx);
                } else {
                    currentRetrainIdx = -1;
                    retrainBoxes = [];
                    currentRetrainImgObj = null;
                    if (rCtx) rCtx.clearRect(0, 0, retrainCanvas.width, retrainCanvas.height);
                    const nameEl = document.getElementById('retrain-filename');
                    if (nameEl) nameEl.innerText = '(No images staged - Empty Project)';
                }
            } catch (err) {
                console.error('Error fetching retrain images:', err);
            }
        }

        function renderRetrainList() {
            const container = document.getElementById('retrain-images-list');
            container.innerHTML = '';
            const annotated = retrainImages.filter(i => i.count > 0).length;
            document.getElementById('retrain-progress-badge').innerText = `${annotated} / ${retrainImages.length}`;

            if (retrainImages.length === 0) {
                container.innerHTML = `
                    <div style="padding: 24px 12px; text-align: center; color: var(--text-muted); font-size: 12px; background: rgba(255,255,255,0.02); border-radius: 6px; border: 1px dashed #2e303d; margin-top: 6px;">
                        <div style="font-size: 26px; margin-bottom: 6px;">📂</div>
                        <div style="font-weight: 600; color: #f0f2f5; margin-bottom: 4px;">Project Dataset Empty</div>
                        <div style="font-size: 11px; line-height: 1.4;">No images in this project yet.<br>Click <b>Upload 10–15 Images</b> above to stage PCB photos for YOLO annotation.</div>
                    </div>
                `;
                return;
            }

            retrainImages.forEach((img, idx) => {
                const div = document.createElement('div');
                div.className = 'retrain-item' + (idx === currentRetrainIdx ? ' active' : '');
                div.onclick = () => {
                    saveRetrainAnnotations(false);
                    loadRetrainImage(idx);
                };

                const nameSpan = document.createElement('span');
                nameSpan.innerText = img.name;

                const tag = document.createElement('span');
                tag.className = 'tag-pill' + (img.count > 0 ? ' done' : '');
                tag.innerText = img.count > 0 ? `${img.count} balls` : 'Pending';

                div.appendChild(nameSpan);
                div.appendChild(tag);
                container.appendChild(div);
            });
        }

        async function handleRetrainUpload(input) {
            if (!input.files || input.files.length === 0) return;
            const formData = new FormData();
            for (let i = 0; i < input.files.length; i++) {
                formData.append('images', input.files[i]);
            }
            const res = await fetch('/api/retrain/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            alert(`Uploaded ${data.count} image(s) for retraining.`);
            fetchRetrainImages();
        }

        async function loadRetrainImage(idx) {
            if (idx < 0 || idx >= retrainImages.length) return;
            currentRetrainIdx = idx;
            const imgData = retrainImages[currentRetrainIdx];
            document.getElementById('retrain-filename').innerText = imgData.name;

            // Load annotations
            const annRes = await fetch(`/api/retrain/annotation/${encodeURIComponent(imgData.name)}`);
            retrainBoxes = await annRes.json();

            // Load image
            retrainImgElement.onload = () => {
                retrainCanvas.width = retrainImgElement.width;
                retrainCanvas.height = retrainImgElement.height;
                fitRetrainToScreen();
                drawRetrain();
            };
            retrainImgElement.src = `/api/retrain/image/${encodeURIComponent(imgData.name)}`;
            renderRetrainList();
        }

        function fitRetrainToScreen() {
            const vw = retrainViewport.clientWidth;
            const vh = retrainViewport.clientHeight;
            const iw = retrainImgElement.width;
            const ih = retrainImgElement.height;
            if (iw === 0 || ih === 0) return;

            rScale = Math.min((vw - 40) / iw, (vh - 40) / ih, 1.0);
            rPanX = (vw - iw * rScale) / 2;
            rPanY = (vh - ih * rScale) / 2;
            updateRetrainTransform();
        }

        function updateRetrainTransform() {
            retrainCanvas.style.transform = `translate(${rPanX}px, ${rPanY}px) scale(${rScale})`;
        }

        function drawRetrain() {
            rCtx.clearRect(0, 0, retrainCanvas.width, retrainCanvas.height);
            rCtx.drawImage(retrainImgElement, 0, 0);

            // Draw boxes
            retrainBoxes.forEach((b, i) => {
                const w = b.x2 - b.x1;
                const h = b.y2 - b.y1;

                rCtx.strokeStyle = '#00ff66';
                rCtx.lineWidth = Math.max(1, 2 / rScale);
                rCtx.strokeRect(b.x1, b.y1, w, h);

                const cx = (b.x1 + b.x2) / 2;
                const cy = (b.y1 + b.y2) / 2;
                rCtx.fillStyle = '#ff2255';
                rCtx.beginPath();
                rCtx.arc(cx, cy, Math.max(2, 3 / rScale), 0, Math.PI * 2);
                rCtx.fill();

                const fontSize = Math.max(11, 14 / rScale);
                rCtx.font = `bold ${fontSize}px sans-serif`;
                rCtx.fillStyle = '#00ff66';
                rCtx.fillRect(b.x1, Math.max(0, b.y1 - fontSize - 4), 30, fontSize + 4);
                rCtx.fillStyle = '#000';
                rCtx.fillText(`#${i+1}`, b.x1 + 3, Math.max(fontSize, b.y1 - 3));
            });

            // Drag box preview
            if (rIsDrawing && retrainMode === 'drag') {
                const x1 = Math.min(rDrawStartX, rDrawCurX);
                const y1 = Math.min(rDrawStartY, rDrawCurY);
                const x2 = Math.max(rDrawStartX, rDrawCurX);
                const y2 = Math.max(rDrawStartY, rDrawCurY);

                rCtx.strokeStyle = '#00ccff';
                rCtx.lineWidth = Math.max(1.5, 2.5 / rScale);
                rCtx.setLineDash([4 / rScale, 4 / rScale]);
                rCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                rCtx.setLineDash([]);
            }
        }

        function getRetrainCanvasCoords(e) {
            const rect = retrainCanvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rScale;
            const y = (e.clientY - rect.top) / rScale;
            return {
                x: Math.max(0, Math.min(retrainCanvas.width, Math.round(x))),
                y: Math.max(0, Math.min(retrainCanvas.height, Math.round(y)))
            };
        }

        retrainViewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.25 : 0.8;
            const newScale = Math.min(Math.max(0.1, rScale * factor), 25.0);
            rPanX = e.clientX - (e.clientX - rPanX) * (newScale / rScale);
            rPanY = e.clientY - (e.clientY - rPanY) * (newScale / rScale);
            rScale = newScale;
            updateRetrainTransform();
            drawRetrain();
        }, { passive: false });

        retrainViewport.addEventListener('mousedown', (e) => {
            if (e.button === 2 || e.button === 1 || e.shiftKey) {
                rIsPanning = true;
                rStartPanX = e.clientX - rPanX;
                rStartPanY = e.clientY - rPanY;
                retrainViewport.style.cursor = 'grab';
                return;
            }
            if (e.button === 0) {
                const coords = getRetrainCanvasCoords(e);
                if (retrainMode === 'click') {
                    const r = quickRadius;
                    retrainBoxes.push({
                        x1: Math.max(0, coords.x - r),
                        y1: Math.max(0, coords.y - r),
                        x2: Math.min(retrainCanvas.width, coords.x + r),
                        y2: Math.min(retrainCanvas.height, coords.y + r)
                    });
                    drawRetrain();
                    saveRetrainAnnotations(false);
                } else {
                    rIsDrawing = true;
                    rDrawStartX = coords.x;
                    rDrawStartY = coords.y;
                    rDrawCurX = coords.x;
                    rDrawCurY = coords.y;
                }
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (rIsPanning) {
                rPanX = e.clientX - rStartPanX;
                rPanY = e.clientY - rStartPanY;
                updateRetrainTransform();
                return;
            }
            if (rIsDrawing) {
                const coords = getRetrainCanvasCoords(e);
                rDrawCurX = coords.x;
                rDrawCurY = coords.y;
                drawRetrain();
            }
        });

        window.addEventListener('mouseup', (e) => {
            if (rIsPanning) {
                rIsPanning = false;
                retrainViewport.style.cursor = 'crosshair';
            }
            if (rIsDrawing) {
                rIsDrawing = false;
                const x1 = Math.min(rDrawStartX, rDrawCurX);
                const y1 = Math.min(rDrawStartY, rDrawCurY);
                const x2 = Math.max(rDrawStartX, rDrawCurX);
                const y2 = Math.max(rDrawStartY, rDrawCurY);
                if (x2 - x1 >= 4 && y2 - y1 >= 4) {
                    retrainBoxes.push({ x1, y1, x2, y2 });
                    saveRetrainAnnotations(false);
                }
                drawRetrain();
            }
        });
        retrainViewport.addEventListener('contextmenu', e => e.preventDefault());

        function setRetrainMode(mode) {
            retrainMode = mode;
            document.getElementById('btn-mode-drag').classList.toggle('active', mode === 'drag');
            document.getElementById('btn-mode-click').classList.toggle('active', mode === 'click');
        }

        function undoRetrainBox() {
            if (retrainBoxes.length > 0) {
                retrainBoxes.pop();
                drawRetrain();
                saveRetrainAnnotations(false);
            }
        }

        async function saveRetrainAnnotations(notify = true) {
            if (!retrainImages[currentRetrainIdx]) return;
            const name = retrainImages[currentRetrainIdx].name;
            await fetch(`/api/retrain/annotation/${encodeURIComponent(name)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ boxes: retrainBoxes })
            });
            retrainImages[currentRetrainIdx].count = retrainBoxes.length;
            renderRetrainList();
            if (notify) alert('Annotations saved!');
        }

        async function runAIPreAnnotate() {
            if (!retrainImages[currentRetrainIdx]) return;
            const name = retrainImages[currentRetrainIdx].name;
            const res = await fetch(`/api/retrain/pre_annotate/${encodeURIComponent(name)}`);
            const suggested = await res.json();
            if (suggested && suggested.length > 0) {
                retrainBoxes = [...retrainBoxes, ...suggested];
                drawRetrain();
                saveRetrainAnnotations(false);
                alert(`AI suggested ${suggested.length} solder ball location(s)!`);
            } else {
                alert('No solder balls detected by AI.');
            }
        }

        async function deleteCurrentRetrainImage() {
            if (!retrainImages[currentRetrainIdx]) return;
            const name = retrainImages[currentRetrainIdx].name;
            if (!confirm(`Delete "${name}" from retraining staging?`)) return;

            await fetch(`/api/retrain/image/${encodeURIComponent(name)}`, { method: 'DELETE' });
            retrainImages.splice(currentRetrainIdx, 1);
            if (currentRetrainIdx >= retrainImages.length) currentRetrainIdx = Math.max(0, retrainImages.length - 1);
            renderRetrainList();
            if (retrainImages.length > 0) loadRetrainImage(currentRetrainIdx);
            else {
                rCtx.clearRect(0, 0, retrainCanvas.width, retrainCanvas.height);
                document.getElementById('retrain-filename').innerText = '(No images left)';
            }
        }

        function prevRetrainImage() {
            if (currentRetrainIdx > 0) {
                saveRetrainAnnotations(false);
                loadRetrainImage(currentRetrainIdx - 1);
            }
        }
        function nextRetrainImage() {
            if (currentRetrainIdx < retrainImages.length - 1) {
                saveRetrainAnnotations(false);
                loadRetrainImage(currentRetrainIdx + 1);
            }
        }

        // ================= RETRAINING PROCESS =================
        let trainPollTimer = null;

        async function startRetraining() {
            const annotatedCount = retrainImages.filter(i => i.count > 0).length;
            if (annotatedCount === 0) {
                return alert('Please annotate at least one image before retraining.');
            }

            const epochs = parseInt(document.getElementById('retrain-epochs-slider').value);
            const weightsSelect = document.getElementById('retrain-weights-select');
            const weightsChoice = weightsSelect ? weightsSelect.value : 'nano';
            const weightsLabel = weightsChoice === 'nano' ? 'from scratch (yolov8n.pt)' : (weightsChoice === 'project' ? 'using project checkpoint' : 'fine-tuning pre-trained model');

            if (!confirm(`Start retraining YOLO model with ${annotatedCount} newly annotated images for ${epochs} epochs ${weightsLabel}?`)) {
                return;
            }

            document.getElementById('btn-start-retrain').disabled = true;
            document.getElementById('train-status-box').style.display = 'block';
            document.getElementById('train-status-title').innerText = 'Initializing Training Pipeline...';

            const res = await fetch('/api/retrain/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ epochs: epochs, weights_choice: weightsChoice })
            });
            const data = await res.json();
            if (data.status !== 'started') {
                alert('Failed to start training: ' + (data.message || 'unknown error'));
                document.getElementById('btn-start-retrain').disabled = false;
                return;
            }

            trainPollTimer = setInterval(pollTrainingStatus, 2000);
        }

        async function pollTrainingStatus() {
            const res = await fetch('/api/retrain/status');
            const state = await res.json();

            document.getElementById('train-percent').innerText = `${state.progress}%`;
            document.getElementById('train-progress-fill').style.width = `${state.progress}%`;

            if (state.logs && state.logs.length > 0) {
                const consoleDiv = document.getElementById('train-console');
                consoleDiv.innerText = state.logs.join('\\n');
                consoleDiv.scrollTop = consoleDiv.scrollHeight;
            }

            if (state.running) {
                document.getElementById('train-status-title').innerText = 
                    `Training: Epoch ${state.current_epoch} / ${state.total_epochs} (Loss: ${state.loss.toFixed(4)})`;
            } else if (state.completed) {
                clearInterval(trainPollTimer);
                document.getElementById('train-status-title').innerText = '🎉 Training Completed Successfully!';
                document.getElementById('btn-start-retrain').disabled = false;
                document.getElementById('model-status-text').innerText = `YOLOv8 Active (Updated: ${state.model_version})`;
                alert('Model retrained successfully! The newly trained weights are now active in the Inspector.');
                fetchRetrainImages();
            } else if (state.error) {
                clearInterval(trainPollTimer);
                document.getElementById('train-status-title').innerText = '❌ Training Failed';
                document.getElementById('btn-start-retrain').disabled = false;
                alert('Training failed: ' + state.error);
            }
        }

        // Keyboard hotkeys for annotation
        window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            if (corrModal.style.display === 'flex') {
                if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) undoCorrBox();
                else if (e.key === '1') setCorrMode('drag');
                else if (e.key === '2') setCorrMode('click');
                else if (e.key === 'Escape') closeCorrectionModal();
                return;
            }
            if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') prevRetrainImage();
            else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') nextRetrainImage();
            else if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) undoRetrainBox();
            else if (e.key === '1') setRetrainMode('drag');
            else if (e.key === '2') setRetrainMode('click');
        });

        // ================= CORRECTION MODAL STATE & LOGIC =================
        let corrBoxes = []; // Array of {x1, y1, x2, y2}
        let corrMode = 'click'; // 'click' or 'drag'
        let corrRadius = 24;
        let corrImgElement = new Image();
        let cScale = 1.0, cPanX = 0, cPanY = 0;
        let cIsPanning = false, cStartPanX = 0, cStartPanY = 0;
        let cIsDrawing = false, cDrawStartX = 0, cDrawStartY = 0, cDrawCurX = 0, cDrawCurY = 0;
        let hoveredBoxIdx = -1;

        const corrModal = document.getElementById('correction-modal');
        const corrViewport = document.getElementById('corr-viewport');
        const corrCanvas = document.getElementById('corr-canvas');
        const cCtx = corrCanvas.getContext('2d');

        function openCorrectionModal() {
            if (!currentInspectImageB64) {
                alert('No image to correct.');
                return;
            }

            // Populate boxes from current detections
            corrBoxes = (currentDetections || []).map(d => ({
                x1: d.box[0],
                y1: d.box[1],
                x2: d.box[2],
                y2: d.box[3]
            }));

            document.getElementById('corr-filename').innerText = currentFilename;
            updateCorrCount();

            corrImgElement.onload = () => {
                corrCanvas.width = corrImgElement.width;
                corrCanvas.height = corrImgElement.height;
                fitCorrToScreen();
                drawCorr();
            };
            corrImgElement.src = currentInspectImageB64;

            corrModal.style.display = 'flex';
        }

        function closeCorrectionModal() {
            corrModal.style.display = 'none';
        }

        function updateCorrCount() {
            document.getElementById('corr-count-badge').innerText = `${corrBoxes.length} Balls`;
        }

        function setCorrMode(mode) {
            corrMode = mode;
            document.getElementById('btn-corr-click').classList.toggle('active', mode === 'click');
            document.getElementById('btn-corr-drag').classList.toggle('active', mode === 'drag');
        }

        function fitCorrToScreen() {
            const vw = corrViewport.clientWidth;
            const vh = corrViewport.clientHeight;
            const iw = corrImgElement.width;
            const ih = corrImgElement.height;
            if (iw === 0 || ih === 0) return;

            cScale = Math.min((vw - 40) / iw, (vh - 40) / ih, 1.0);
            cPanX = (vw - iw * cScale) / 2;
            cPanY = (vh - ih * cScale) / 2;
            updateCorrTransform();
        }

        function updateCorrTransform() {
            corrCanvas.style.transform = `translate(${cPanX}px, ${cPanY}px) scale(${cScale})`;
            document.getElementById('corr-zoom-text').innerText = `${Math.round(cScale * 100)}%`;
        }

        function drawCorr() {
            cCtx.clearRect(0, 0, corrCanvas.width, corrCanvas.height);
            cCtx.drawImage(corrImgElement, 0, 0);

            // Draw boxes
            corrBoxes.forEach((b, i) => {
                const isHovered = (i === hoveredBoxIdx);
                const w = b.x2 - b.x1;
                const h = b.y2 - b.y1;

                cCtx.strokeStyle = isHovered ? '#ff2255' : '#00ff66';
                cCtx.lineWidth = Math.max(1.5, (isHovered ? 3 : 2) / cScale);
                cCtx.strokeRect(b.x1, b.y1, w, h);

                const cx = (b.x1 + b.x2) / 2;
                const cy = (b.y1 + b.y2) / 2;
                cCtx.fillStyle = isHovered ? '#ff2255' : '#00ff66';
                cCtx.beginPath();
                cCtx.arc(cx, cy, Math.max(2, 3 / cScale), 0, Math.PI * 2);
                cCtx.fill();

                const fontSize = Math.max(11, 14 / cScale);
                cCtx.font = `bold ${fontSize}px sans-serif`;
                const labelText = isHovered ? `✕ Delete #${i+1}` : `#${i+1}`;
                const tw = cCtx.measureText(labelText).width;

                cCtx.fillStyle = isHovered ? '#ff2255' : '#00ff66';
                cCtx.fillRect(b.x1, Math.max(0, b.y1 - fontSize - 4), tw + 6, fontSize + 4);
                cCtx.fillStyle = isHovered ? '#ffffff' : '#000000';
                cCtx.fillText(labelText, b.x1 + 3, Math.max(fontSize, b.y1 - 3));
            });

            // Drawing box preview
            if (cIsDrawing && corrMode === 'drag') {
                const x1 = Math.min(cDrawStartX, cDrawCurX);
                const y1 = Math.min(cDrawStartY, cDrawCurY);
                const x2 = Math.max(cDrawStartX, cDrawCurX);
                const y2 = Math.max(cDrawStartY, cDrawCurY);

                cCtx.strokeStyle = '#00ccff';
                cCtx.lineWidth = Math.max(1.5, 2.5 / cScale);
                cCtx.setLineDash([4 / cScale, 4 / cScale]);
                cCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                cCtx.setLineDash([]);
            }
        }

        function getCorrCanvasCoords(e) {
            const rect = corrCanvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / cScale;
            const y = (e.clientY - rect.top) / cScale;
            return {
                x: Math.max(0, Math.min(corrCanvas.width, Math.round(x))),
                y: Math.max(0, Math.min(corrCanvas.height, Math.round(y)))
            };
        }

        corrViewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.25 : 0.8;
            const newScale = Math.min(Math.max(0.1, cScale * factor), 25.0);
            cPanX = e.clientX - (e.clientX - cPanX) * (newScale / cScale);
            cPanY = e.clientY - (e.clientY - cPanY) * (newScale / cScale);
            cScale = newScale;
            updateCorrTransform();
            drawCorr();
        }, { passive: false });

        corrViewport.addEventListener('mousedown', (e) => {
            if (e.button === 2 || e.button === 1 || e.shiftKey) {
                cIsPanning = true;
                cStartPanX = e.clientX - cPanX;
                cStartPanY = e.clientY - cPanY;
                corrViewport.style.cursor = 'grab';
                return;
            }

            if (e.button === 0) {
                // If clicked on a hovered box -> delete that box!
                if (hoveredBoxIdx !== -1) {
                    corrBoxes.splice(hoveredBoxIdx, 1);
                    hoveredBoxIdx = -1;
                    updateCorrCount();
                    drawCorr();
                    return;
                }

                const coords = getCorrCanvasCoords(e);
                if (corrMode === 'click') {
                    const r = corrRadius;
                    corrBoxes.push({
                        x1: Math.max(0, coords.x - r),
                        y1: Math.max(0, coords.y - r),
                        x2: Math.min(corrCanvas.width, coords.x + r),
                        y2: Math.min(corrCanvas.height, coords.y + r)
                    });
                    updateCorrCount();
                    drawCorr();
                } else {
                    cIsDrawing = true;
                    cDrawStartX = coords.x;
                    cDrawStartY = coords.y;
                    cDrawCurX = coords.x;
                    cDrawCurY = coords.y;
                }
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (corrModal.style.display !== 'flex') return;

            if (cIsPanning) {
                cPanX = e.clientX - cStartPanX;
                cPanY = e.clientY - cStartPanY;
                updateCorrTransform();
                return;
            }

            if (cIsDrawing) {
                const coords = getCorrCanvasCoords(e);
                cDrawCurX = coords.x;
                cDrawCurY = coords.y;
                drawCorr();
                return;
            }

            // Check if hovering over any existing box
            const coords = getCorrCanvasCoords(e);
            let foundHover = -1;
            for (let i = corrBoxes.length - 1; i >= 0; i--) {
                const b = corrBoxes[i];
                if (coords.x >= b.x1 && coords.x <= b.x2 && coords.y >= b.y1 && coords.y <= b.y2) {
                    foundHover = i;
                    break;
                }
            }

            if (foundHover !== hoveredBoxIdx) {
                hoveredBoxIdx = foundHover;
                corrViewport.style.cursor = (hoveredBoxIdx !== -1) ? 'pointer' : 'crosshair';
                drawCorr();
            }
        });

        window.addEventListener('mouseup', (e) => {
            if (corrModal.style.display !== 'flex') return;

            if (cIsPanning) {
                cIsPanning = false;
                corrViewport.style.cursor = 'crosshair';
            }

            if (cIsDrawing) {
                cIsDrawing = false;
                const x1 = Math.min(cDrawStartX, cDrawCurX);
                const y1 = Math.min(cDrawStartY, cDrawCurY);
                const x2 = Math.max(cDrawStartX, cDrawCurX);
                const y2 = Math.max(cDrawStartY, cDrawCurY);
                if (x2 - x1 >= 4 && y2 - y1 >= 4) {
                    corrBoxes.push({ x1, y1, x2, y2 });
                    updateCorrCount();
                }
                drawCorr();
            }
        });

        corrViewport.addEventListener('contextmenu', e => e.preventDefault());

        function undoCorrBox() {
            if (corrBoxes.length > 0) {
                corrBoxes.pop();
                updateCorrCount();
                drawCorr();
            }
        }

        function clearCorrBoxes() {
            if (confirm('Clear all detected boxes for this image?')) {
                corrBoxes = [];
                updateCorrCount();
                drawCorr();
            }
        }

        async function submitCorrection(action) {
            const count = corrBoxes.length;
            const res = await fetch('/api/correction/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_b64: currentInspectImageB64,
                    filename: currentFilename,
                    boxes: corrBoxes,
                    action: action
                })
            });
            const data = await res.json();

            closeCorrectionModal();

            if (action === 'queue') {
                alert(`✅ Correction saved!\n\n"${data.filename}" has been added to the Retraining Studio with ${count} verified solder ball(s).\n\nYou can retrain anytime under the 'Retrain Model' tab.`);
            } else if (action === 'retrain_now') {
                switchView('retrain');
                document.getElementById('btn-start-retrain').disabled = true;
                document.getElementById('train-status-box').style.display = 'block';
                document.getElementById('train-status-title').innerText = 'Fine-tuning with your corrected detection...';
                trainPollTimer = setInterval(pollTrainingStatus, 2000);
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            loadSolderProjects();
        });
    </script>
</body>
</html>
"""

# ================= BACKEND REST API =================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect', methods=['POST'])
def detect_solder_balls():
    data = request.json or {}
    image_b64 = data.get('image_b64', '')
    conf_thresh = float(data.get('conf', 0.25))
    iou_thresh = float(data.get('iou', 0.45))
    filename = data.get('filename', 'pcb.jpg')

    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64
    if ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]
    img_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    h, w = img.shape[:2]
    model = get_or_load_model()

    start_time = time.time()
    detections = []

    if model is not None:
        results = model.predict(img, conf=conf_thresh, iou=iou_thresh, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': round(conf, 3)
                })
    else:
        # Fallback CV circle detector
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
            param1=50, param2=28, minRadius=3, maxRadius=25
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                x, y, r = c[0], c[1], c[2]
                detections.append({
                    'box': [int(x - r), int(y - r), int(x + r), int(y + r)],
                    'conf': 0.85
                })

    inf_time_ms = int((time.time() - start_time) * 1000)

    # Draw neon annotations on image
    annotated = img.copy()
    count = len(detections)

    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['box']
        conf = det['conf']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Highlight box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Center red dot
        cv2.circle(annotated, ((x1 + x2) // 2, (y1 + y2) // 2), 2, (0, 0, 255), -1)

        # Badge tag
        lbl = f"#{idx} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lbl_y1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, lbl_y1), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(annotated, lbl, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Header banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (20, 20, 24), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    status_str = f"SOLDER BALLS FOUND: {count}"
    status_col = (0, 255, 120)
    cv2.putText(annotated, status_str, (18, 33), cv2.FONT_HERSHEY_DUPLEX, 0.85, status_col, 2)
    cv2.putText(annotated, f"File: {filename}", (w - 380, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Encode back to JPEG base64
    _, enc_img = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(enc_img).decode('utf-8')

    return jsonify({
        'count': count,
        'detections': detections,
        'inference_time_ms': inf_time_ms,
        'annotated_b64': annotated_b64
    })

# @app.route('/api/sample_images')
# def list_sample_images():
#     paths = get_solder_paths()
#     files = sorted(glob.glob(os.path.join(paths['dataset_dir'], '*.jpg')) +
#                    glob.glob(os.path.join(paths['dataset_dir'], '*.png')))
#     if not files and not paths['is_default']:
#         def_paths = get_solder_paths('default')
#         files = sorted(glob.glob(os.path.join(def_paths['dataset_dir'], '*.jpg')) +
#                        glob.glob(os.path.join(def_paths['dataset_dir'], '*.png')))
#     return jsonify([os.path.basename(f) for f in files[:40]])

@app.route('/api/sample_images')
def list_sample_images():
    files = sorted(
        glob.glob(os.path.join(SAMPLE_DATASET_DIR, '*.jpg')) +
        glob.glob(os.path.join(SAMPLE_DATASET_DIR, '*.png'))
    )
    return jsonify([os.path.basename(f) for f in files[:40]])

# @app.route('/api/sample_image/<path:filename>')
# def get_sample_image(filename):
#     paths = get_solder_paths()
#     fpath = os.path.join(paths['dataset_dir'], filename)
#     if not os.path.exists(fpath):
#         def_paths = get_solder_paths('default')
#         fpath = os.path.join(def_paths['dataset_dir'], filename)
#     if os.path.exists(fpath):
#         return send_file(fpath)
#     return "Not found", 404

@app.route('/api/sample_image/<path:filename>')
def get_sample_image(filename):
    # Load sample images from the bundled dataset folder
    fpath = os.path.join(SAMPLE_DATASET_DIR, filename)

    if os.path.exists(fpath):
        return send_file(fpath)

    return "Not found", 404

# ================= PROJECT MANAGEMENT API =================

@app.route('/api/retrain/projects', methods=['GET'])
def api_solder_projects_list():
    return jsonify({
        'active_id': ACTIVE_SOLDER_PROJECT_ID,
        'projects': list_solder_projects()
    })

@app.route('/api/retrain/projects/create', methods=['POST'])
def api_solder_projects_create():
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'status': 'error', 'message': 'Project name is required'}), 400
    meta = create_solder_project(name)
    return jsonify({'status': 'ok', 'project': meta})

@app.route('/api/retrain/projects/select', methods=['POST'])
def api_solder_projects_select():
    global ACTIVE_SOLDER_PROJECT_ID
    data = request.get_json(silent=True) or {}
    pid = data.get('project_id', '').strip()
    if not pid:
        return jsonify({'status': 'error', 'message': 'Project ID is required'}), 400
    
    if pid != 'default':
        p_dir = os.path.join(SOLDER_PROJECTS_DIR, pid)
        if not os.path.exists(p_dir):
            return jsonify({'status': 'error', 'message': 'Project not found'}), 404
            
    ACTIVE_SOLDER_PROJECT_ID = pid
    get_or_load_model(pid)
    return jsonify({'status': 'ok', 'active_id': ACTIVE_SOLDER_PROJECT_ID})

@app.route('/api/retrain/projects/delete', methods=['POST'])
def api_solder_projects_delete():
    data = request.get_json(silent=True) or {}
    pid = data.get('project_id', '').strip()
    ok, msg = delete_solder_project(pid)
    if not ok:
        return jsonify({'status': 'error', 'message': msg}), 400
    return jsonify({'status': 'ok', 'message': msg, 'active_id': ACTIVE_SOLDER_PROJECT_ID})

# ================= RETRAINING API =================

@app.route('/api/retrain/upload', methods=['POST'])
def upload_retrain_images():
    paths = get_solder_paths()
    uploaded = request.files.getlist('images')
    saved_count = 0
    for f in uploaded:
        if f.filename:
            fname = os.path.basename(f.filename)
            dst = os.path.join(paths['retrain_img_dir'], fname)
            f.save(dst)
            saved_count += 1
    return jsonify({'status': 'ok', 'count': saved_count})

@app.route('/api/retrain/list')
def list_retrain_images():
    paths = get_solder_paths()
    files = sorted(glob.glob(os.path.join(paths['retrain_img_dir'], '*.jpg')) +
                   glob.glob(os.path.join(paths['retrain_img_dir'], '*.png')) +
                   glob.glob(os.path.join(paths['retrain_img_dir'], '*.jpeg')))
    result = []
    for f in files:
        bname = os.path.basename(f)
        bnoext = os.path.splitext(bname)[0]
        ann_path = os.path.join(paths['retrain_lbl_dir'], f"{bnoext}.txt")
        count = 0
        if os.path.exists(ann_path):
            with open(ann_path, 'r', encoding='utf-8') as af:
                lines = [l.strip() for l in af if l.strip()]
                count = len(lines)
        result.append({'name': bname, 'count': count})
    return jsonify(result)

@app.route('/api/retrain/image/<path:filename>')
def get_retrain_image(filename):
    paths = get_solder_paths()
    fpath = os.path.join(paths['retrain_img_dir'], filename)
    if os.path.exists(fpath):
        return send_file(fpath)
    return "Not found", 404

@app.route('/api/retrain/annotation/<path:filename>', methods=['GET'])
def get_retrain_annotation(filename):
    paths = get_solder_paths()
    bnoext = os.path.splitext(filename)[0]
    ann_path = os.path.join(paths['retrain_lbl_dir'], f"{bnoext}.txt")
    img_path = os.path.join(paths['retrain_img_dir'], filename)

    boxes = []
    if os.path.exists(ann_path) and os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            ih, iw = img.shape[:2]
            with open(ann_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = map(float, parts[1:5])
                        x1 = int(round((xc - w / 2.0) * iw))
                        y1 = int(round((yc - h / 2.0) * ih))
                        x2 = int(round((xc + w / 2.0) * iw))
                        y2 = int(round((yc + h / 2.0) * ih))
                        boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    return jsonify(boxes)

@app.route('/api/retrain/annotation/<path:filename>', methods=['POST'])
def save_retrain_annotation(filename):
    paths = get_solder_paths()
    bnoext = os.path.splitext(filename)[0]
    ann_path = os.path.join(paths['retrain_lbl_dir'], f"{bnoext}.txt")
    img_path = os.path.join(paths['retrain_img_dir'], filename)

    data = request.json or {}
    boxes = data.get('boxes', [])

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            ih, iw = img.shape[:2]
            lines = []
            for b in boxes:
                x1, y1 = max(0, b['x1']), max(0, b['y1'])
                x2, y2 = min(iw, b['x2']), min(ih, b['y2'])
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:
                    xc = (x1 + x2) / (2.0 * iw)
                    yc = (y1 + y2) / (2.0 * ih)
                    lines.append(f"0 {xc:.6f} {yc:.6f} {(w/iw):.6f} {(h/ih):.6f}")

            with open(ann_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    return jsonify({'status': 'ok', 'count': len(boxes)})

@app.route('/api/retrain/image/<path:filename>', methods=['DELETE'])
def delete_retrain_image(filename):
    paths = get_solder_paths()
    img_path = os.path.join(paths['retrain_img_dir'], filename)
    bnoext = os.path.splitext(filename)[0]
    ann_path = os.path.join(paths['retrain_lbl_dir'], f"{bnoext}.txt")
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists(ann_path):
        os.remove(ann_path)
    return jsonify({'status': 'ok'})

@app.route('/api/retrain/pre_annotate/<path:filename>')
def pre_annotate_retrain_image(filename):
    paths = get_solder_paths()
    img_path = os.path.join(paths['retrain_img_dir'], filename)
    if not os.path.exists(img_path):
        return jsonify([])

    img = cv2.imread(img_path)
    if img is None:
        return jsonify([])

    model = get_or_load_model()
    boxes = []
    if model is not None:
        results = model.predict(img, conf=0.25, verbose=False)
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                boxes.append({
                    'x1': int(x1), 'y1': int(y1),
                    'x2': int(x2), 'y2': int(y2)
                })
    return jsonify(boxes)

# ================= BACKGROUND RETRAINING WORKER =================

def background_retrain_task(epochs, batch_size, weights_choice="nano", project_id=None):
    global training_state, CURRENT_MODEL
    paths = get_solder_paths(project_id)

    training_state['running'] = True
    training_state['completed'] = False
    training_state['error'] = None
    training_state['progress'] = 0
    training_state['current_epoch'] = 0
    training_state['total_epochs'] = epochs
    training_state['loss'] = 0.0
    training_state['logs'] = [f"Starting active retraining pipeline for project '{paths['id']}'..."]

    try:
        # 1. Merge retrain images and annotations into dataset/ and annotations/
        staged_imgs = glob.glob(os.path.join(paths['retrain_img_dir'], '*.*'))
        merged_count = 0
        for img_f in staged_imgs:
            bname = os.path.basename(img_f)
            bnoext = os.path.splitext(bname)[0]
            lbl_f = os.path.join(paths['retrain_lbl_dir'], f"{bnoext}.txt")
            if os.path.exists(lbl_f):
                shutil.copy2(img_f, os.path.join(paths['dataset_dir'], bname))
                shutil.copy2(lbl_f, os.path.join(paths['annotations_dir'], f"{bnoext}.txt"))
                merged_count += 1

        training_state['logs'].append(f"Merged {merged_count} new annotated images into project dataset.")

        # 2. Run internal dataset splitter
        training_state['logs'].append("Rebuilding train / validation splits...")
        ok, msg = split_solder_dataset_internal(paths, val_ratio=0.2)
        if not ok:
            raise RuntimeError(msg)
        training_state['logs'].append(msg)

        # 3. Load model base weights
        yolo_base = os.path.join(WORKSPACE_DIR, 'yolov8n.pt')
        if weights_choice == 'project' and os.path.exists(paths['model_path']):
            base_weights = paths['model_path']
        elif weights_choice == 'best' and os.path.exists(MODEL_PATH):
            base_weights = MODEL_PATH
        elif os.path.exists(paths['model_path']):
            base_weights = paths['model_path']
        elif os.path.exists(yolo_base):
            base_weights = yolo_base
        else:
            base_weights = 'yolov8n.pt'

        training_state['logs'].append(f"Base weights: {os.path.basename(base_weights)}")
        model = YOLO(base_weights)

        # Callbacks to monitor progress
        def on_train_epoch_end(trainer):
            ep = trainer.epoch + 1
            tot = trainer.epochs
            training_state['current_epoch'] = ep
            training_state['total_epochs'] = tot
            pct = int((ep / tot) * 100)
            training_state['progress'] = pct
            loss_val = float(trainer.loss.item()) if hasattr(trainer, 'loss') and trainer.loss is not None else 0.0
            training_state['loss'] = loss_val
            msg = f"Epoch {ep}/{tot} | Loss: {loss_val:.4f}"
            training_state['logs'].append(msg)

        def on_fit_epoch_end(trainer):
            metrics = getattr(trainer, 'metrics', {})
            map_val = metrics.get('metrics/mAP50(B)', 0.0)
            training_state['map50'] = round(map_val * 100, 1)

        model.add_callback('on_train_epoch_end', on_train_epoch_end)
        model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

        results = model.train(
            data=paths['data_yaml_path'],
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device='cpu',
            workers=2,
            patience=10,
            project=paths['runs_dir'],
            name='solderball_retrain',
            exist_ok=True
        )

        # Copy best weights to project model path
        best_pt = os.path.join(results.save_dir, 'weights', 'best.pt')
        if os.path.exists(best_pt):
            os.makedirs(os.path.dirname(paths['model_path']), exist_ok=True)
            shutil.copy2(best_pt, paths['model_path'])
            # If default project, also update MODEL_PATH
            if paths['is_default']:
                shutil.copy2(best_pt, MODEL_PATH)

        # Hot-reload in memory
        with MODEL_LOCK:
            CURRENT_MODEL = YOLO(paths['model_path'])
            CURRENT_MODEL._loaded_path = paths['model_path']

        training_state['model_version'] = time.strftime("%Y-%m-%d %H:%M")
        training_state['progress'] = 100
        training_state['completed'] = True
        training_state['running'] = False
        training_state['logs'].append("Retraining completed successfully! Model hot-reloaded.")

    except Exception as e:
        training_state['running'] = False
        training_state['error'] = str(e)
        training_state['logs'].append(f"Error during retraining: {e}")

@app.route('/api/retrain/start', methods=['POST'])
def start_retrain_api():
    if training_state['running']:
        return jsonify({'status': 'busy', 'message': 'Training is already running!'}), 400

    data = request.json or {}
    epochs = int(data.get('epochs', 20))
    batch_size = int(data.get('batch_size', 8))
    weights_choice = data.get('weights_choice', 'nano')
    project_id = ACTIVE_SOLDER_PROJECT_ID

    t = threading.Thread(target=background_retrain_task, args=(epochs, batch_size, weights_choice, project_id), daemon=True)
    t.start()

    return jsonify({'status': 'started', 'epochs': epochs})

@app.route('/api/retrain/status')
def get_retrain_status():
    return jsonify(training_state)

@app.route('/api/correction/save', methods=['POST'])
def save_correction_api():
    paths = get_solder_paths()
    data = request.json or {}
    image_b64 = data.get('image_b64', '')
    filename = data.get('filename', 'corrected_pcb.jpg')
    boxes = data.get('boxes', [])
    action = data.get('action', 'queue')

    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        filename += '.jpg'

    base_no_ext = os.path.splitext(os.path.basename(filename))[0]
    target_img_name = f"corr_{base_no_ext}.jpg"
    target_lbl_name = f"corr_{base_no_ext}.txt"

    dst_img_path = os.path.join(paths['retrain_img_dir'], target_img_name)
    dst_lbl_path = os.path.join(paths['retrain_lbl_dir'], target_lbl_name)

    # 1. Save or copy image file
    if image_b64 and ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]

    if image_b64:
        img_bytes = base64.b64decode(image_b64)
        with open(dst_img_path, 'wb') as f:
            f.write(img_bytes)
    elif os.path.exists(os.path.join(paths['dataset_dir'], filename)):
        shutil.copy2(os.path.join(paths['dataset_dir'], filename), dst_img_path)

    # 2. Convert and save normalized YOLO coordinates
    img = cv2.imread(dst_img_path)
    if img is not None:
        ih, iw = img.shape[:2]
        lines = []
        for b in boxes:
            x1, y1 = max(0, b['x1']), max(0, b['y1'])
            x2, y2 = min(iw, b['x2']), min(ih, b['y2'])
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                xc = (x1 + x2) / (2.0 * iw)
                yc = (y1 + y2) / (2.0 * ih)
                lines.append(f"0 {xc:.6f} {yc:.6f} {(w/iw):.6f} {(h/ih):.6f}")

        with open(dst_lbl_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    # 3. If action is 'retrain_now', launch immediate fine-tuning!
    if action == 'retrain_now':
        if not training_state['running']:
            epochs = int(data.get('epochs', 20))
            t = threading.Thread(target=background_retrain_task, args=(epochs, 8, 'nano', ACTIVE_SOLDER_PROJECT_ID), daemon=True)
            t.start()
            return jsonify({'status': 'retraining_started', 'filename': target_img_name, 'count': len(boxes)})

    return jsonify({'status': 'queued', 'filename': target_img_name, 'count': len(boxes)})

if __name__ == '__main__':
    print("=" * 70)
    print("  SOLDER BALL INSPECTION & RETRAINING PLATFORM")
    print("=" * 70)
    print("  Opening browser at: http://127.0.0.1:5000")
    print("  Ready to inspect PCB images and retrain custom models.")
    print("=" * 70)

    try:
        webbrowser.open('http://127.0.0.1:5000')
    except Exception:
        pass

    app.run(host='127.0.0.1', port=5000, debug=False)
