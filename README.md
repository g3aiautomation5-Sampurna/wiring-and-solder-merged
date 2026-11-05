# Unified Vision Quality Inspection Platform (Wiring & Soldering)

This platform seamlessly merges Automated Optical Inspection (AOI) utilities into a unified web dashboard:
1. **🔌 Wiring Inspection (Pose Estimation & Topological Verification)**
2. **🔬 Soldering Inspection (Solder Ball Detection, Counting & Active Retraining)**
3. **⚡ Combined Inspection (All-in-One Dual Wiring & Solder Ball Inspection)**

All utilities are accessible via top-level master tabs, preserving 100% of their original functionalities, sub-tabs, cameras, annotations, and retraining capabilities, while offering a unified inspection station where a single Top View image/camera is simultaneously evaluated for wire harness pose and solder ball defects alongside Side View Origin 3 detection.

---

## 🚀 Quick Start

Double-click **[`start_merged_platform.bat`](file:///c:/Users/SM%20461/Desktop/merged%20utility/start_merged_platform.bat)** or run from terminal:

```bash
python merged_app.py --port 5000
```

The browser will automatically open:
```
http://localhost:5000
```

---

## 🧭 Navigation & Shortcuts

- **Wiring Tab**: Click **"Wiring Inspection"** or press `Alt + 1`
- **Soldering Tab**: Click **"Soldering Inspection"** or press `Alt + 2`
- **Combined Tab**: Click **"Combined Inspection"** or press `Alt + 3`
- **Fullscreen**: Click the **"Fullscreen"** button in the header or press `F11`

Direct URLs:
- Master Suite: `http://localhost:5000`
- Direct Wiring: `http://localhost:5000/wiring`
- Direct Soldering: `http://localhost:5000/soldering`
- Direct Combined: `http://localhost:5000/combined`

---

## 🌟 Platform Capabilities

### 1. Combined Inspection (`/combined`) &mdash; *New All-in-One Tab*
- **Shared Top View Evaluation**:
  - The Top Camera / image is used for **BOTH**:
    1. **Wire Harness Pose Inspection**: Evaluates topological routing, wire origins, socket/terminal positions, and wire colors (Upper Middle, Below Left, Below Right).
    2. **Solder Ball Defect Detection**: Runs YOLOv8 solder detector on the exact same board to detect, count, and evaluate solder balls.
- **Side View Origin 3 Detection**:
  - The Side Camera / image is evaluated for Origin 3 component presence, bounding box, and `(cx, cy)` center alignment.
- **Master Unified Quality Verdict**:
  - Overall system **PASS** ONLY if Top Wire Pose passes, Side Origin 3 passes, and Solder Ball evaluation passes.
- **Interactive Multi-Mode Visual Viewports**:
  - **Top View Toggle**: `🔀 Combined Overlay` (Shows wire pose skeleton/labels + solder ball bounding boxes together), `🔌 Wire Pose`, `🔬 Solder Balls`, `📷 Raw Top Image`.
  - **Side View Toggle**: `📐 Origin 3 Detection`, `📷 Raw Side Image`.
  - Zoom & Pan controls and high-res annotated downloads.
- **Dual Camera Live Station & Shutter**:
  - Select and stream live from Camera 1 (Top) and Camera 2 (Side), with a single-click *"Snap & Inspect Both Cameras"* shutter.
  - Drag-and-drop or file upload for Top and Side board images.

### 2. Wiring Inspection (`/wiring`)
- **Dual-Camera & Dual-Image Inspection Station**:
  - Connects Cam 1 (Top Pose: Wire Harness) and Cam 2 (Side View: Origin 3 Object) simultaneously with shutter snap.
  - Drag and drop Top View and Side View images.
  - Component breakdown: Upper Middle (UM), Below Left (BL), Below Right (BR) wires, plus Origin 3 side-view component card.
- **Annotation & Retraining Studio**: Interactive keypoint annotator (3-point clicker: Origin, Midpoint, Terminal), color class palette management, raw image uploads, live epoch loss progress bar, and automatic hot-reloading of trained weights.

### 3. Soldering Inspection (`/soldering`)
- **Inspect & Count**: Live microscope/webcam camera stream, file upload, dataset gallery selector (167+ PCB boards), confidence & IoU threshold sliders, neon green detection boxes with center markers, high-res downloads, and CSV reporting.
- **Human-in-the-Loop Error Correction**: Click *"Wrong Detection? Correct It"* to remove false positives, add missed solder balls, and queue or immediately fine-tune the model.
- **Active Retraining Studio**: Multi-image batch upload, 25× wheel zoom & pan canvas, quick-click marker mode, AI pre-annotate assist, and live retraining progress monitoring.

---

## 📁 File Structure

```
merged utility/
├── merged_app.py               # Main unified WSGI application with master navigation
├── combined_app.py             # All-in-One Dual Wiring & Solder Ball Inspection module
├── start_merged_platform.bat   # 1-Click launcher script (Alt+1, Alt+2, Alt+3)
├── pose_estimation_poc/        # Complete Wire Harness utility
│   ├── web_app.py
│   ├── inspect_pose.py
│   ├── dual_inspection.py
│   ├── data.yaml
│   └── test_samples/
└── solderball detection platform/ # Complete Solder Ball utility
    ├── web_platform.py
    ├── models/
    ├── dataset/
    └── scripts/
```
