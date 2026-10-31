# Unified Vision Quality Inspection Platform (Wiring & Soldering)

This platform seamlessly merges two Automated Optical Inspection (AOI) utilities into a unified web dashboard:
1. **🔌 Wiring Inspection (Pose Estimation & Topological Verification)**
2. **🔬 Soldering Inspection (Solder Ball Detection, Counting & Active Retraining)**

Both utilities are accessible via top-level master tabs, preserving 100% of their original functionalities, sub-tabs, cameras, annotations, and retraining capabilities.

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
- **Fullscreen**: Click the **"Fullscreen"** button in the header or press `F11`

---

## 🌟 Preserved Features in Both Utilities

### 1. Wiring Inspection (`/wiring`)
- **Dual-Camera & Dual-Image Inspection Station**:
  - **Live Dual Cameras**: Connects Cam 1 (Top Pose: Wire Harness) and Cam 2 (Side View: Origin 3 Object) simultaneously with shutter snap.
  - **Dual Upload**: Drag and drop both Top View and Side View images.
  - **1-Click Dual Test Pairs**: Instantly test reference pairs (`✔ Both Pass Pair`, `✘ Top Pose Fail (BL)`, `✘ Origin 3 Missing`).
  - **Comprehensive Component Breakdown**: Evaluates Upper Middle (UM), Below Left (BL), Below Right (BR) wires, plus Origin 3 side-view component card.
  - **Combined Overall Banner**: System passes ONLY if BOTH Top Wire Pose AND Side Origin 3 pass.
- **Annotation & Retraining Studio**: Interactive keypoint annotator (3-point clicker: Origin, Midpoint, Terminal), color class palette management, raw image uploads, live epoch loss progress bar, and automatic hot-reloading of trained weights.

### 2. Soldering Inspection (`/soldering`)
- **Inspect & Count**: Live microscope/webcam camera stream, file upload, dataset gallery selector (167+ PCB boards), confidence & IoU threshold sliders, neon green detection boxes with center markers, high-res downloads, and CSV reporting.
- **Human-in-the-Loop Error Correction**: Click *"Wrong Detection? Correct It"* to remove false positives, add missed solder balls, and queue or immediately fine-tune the model.
- **Active Retraining Studio**: Multi-image batch upload, 25× wheel zoom & pan canvas, quick-click marker mode, AI pre-annotate assist, and live retraining progress monitoring.

---

## 📁 File Structure

```
merged utility/
├── merged_app.py               # Main unified WSGI application with master navigation
├── start_merged_platform.bat   # 1-Click launcher script
├── pose_estimation_poc/        # Complete Wire Harness utility (intact)
│   ├── web_app.py
│   ├── inspect_pose.py
│   ├── data.yaml
│   └── test_samples/
└── solderball detection platform/ # Complete Solder Ball utility (intact)
    ├── web_platform.py
    ├── models/
    ├── dataset/
    └── scripts/
```
