# Solder Ball Inspection & Active Retraining Platform

An Automated Optical Inspection (AOI) platform for detecting and counting solder balls on printed circuit boards (PCBs), with continuous active retraining built-in.

---

## 🚀 Quick Start (Browser-Based Web Platform)

Double-click **`start_platform.bat`** (or run `python web_platform.py`).

Your default web browser will automatically open:
```
http://127.0.0.1:5000
```

---

## 🌟 Platform Capabilities

### 1. 🔍 Solder Ball Inspection & Counting
- **Multiple Input Sources**:
  - **📁 File Upload**: Drag and drop any PCB image (JPG, PNG).
  - **📸 Live Camera**: Click *"Open Live Camera"* to stream from a webcam or USB microscope and snap photos directly into the detector.
  - **🖼️ Dataset Gallery**: Pick any of your existing 167 PCB inspection boards for instant testing.
- **Instant Detection & Counting**:
  - Highlights all detected solder balls with neon green boxes and center markers.
  - Displays exact count: **`Solder Balls Found: N`**.
  - Live sensitivity controls (Confidence & IoU sliders).
  - Download high-res annotated images or export inspection logs as **CSV**.
- **✏️ Interactive Error Correction (Human-in-the-Loop)**:
  - If the model makes a mistake, click **`✏️ Wrong Detection? Correct It`**.
  - All predicted boxes are pre-loaded on the canvas.
  - **Remove false detections**: Hover over any wrong box (turns red) and click to delete it.
  - **Add missed balls**: Click directly on missed solder balls to place a box.
  - **`💾 Save to Retrain Queue`**: Queues the corrected image into the Retraining Studio.
  - **`🚀 Save & Retrain Now`**: Immediately saves and starts fine-tuning the model in the background to learn from the mistake!

### 2. 🔄 Active Retraining Studio
Allows adding new PCB variations or lighting conditions to continuously improve accuracy:
1. **Upload 10–15 Images**: Drag and drop multiple PCB photos into the staging area.
2. **In-Browser Annotation**:
   - Ultra-smooth 60fps canvas with 25× wheel zoom and pan.
   - **🎯 Quick-Click Marker Mode (`2`)**: Click on solder balls to instantly box them.
   - **✨ AI Pre-Annotate**: Generates initial candidate boxes using the current model so you only need to review and adjust!
3. **One-Click Retraining**:
   - Choose epoch count (e.g., 20 epochs) and click **"🚀 Start Retraining"**.
   - Watch real-time progress bar, epoch loss, and live terminal training logs.
   - The newly fine-tuned model is automatically saved to `models/best_solderball_detector.pt` and hot-reloaded into the Inspector without restarting!

---

## 📁 Project Structure

```
solderball detection platform/
├── dataset/                  # 174 raw PCB inspection images
├── data/                     # Formatted YOLO dataset
│   ├── images/train/, val/   # Split training and validation images
│   └── labels/train/, val/   # Split training and validation labels
├── annotations/              # Label files (.txt, .json, or .xml)
├── models/                   # Saved model weights
│   └── best_solderball_detector.pt
├── scripts/
│   ├── split_dataset.py      # Splits annotations into train/val and converts formats
│   ├── verify_annotations.py # Renders bounding boxes to visually verify annotations
│   ├── train.py              # Trains YOLOv8/YOLO11 detector
│   ├── detect_and_count.py   # CLI inference script: detects, counts, and logs results
│   └── auto_annotate.py      # Active learning script: auto-labels remaining images
├── app.py                    # Streamlit Web App: Interactive UI for detection & counting
└── data.yaml                 # YOLO dataset configuration
```

---

## 🚀 Step-by-Step Workflow

### Step 1: Annotate Images (100% Offline)

You have two offline annotators ready in this workspace:

#### Option 1: Browser-Based Local Annotator (Recommended)
Run:
```bash
python annotate_tool.py
```
*(or double-click `start_annotator.bat`)*

- Runs locally at `http://127.0.0.1:5000` (100% offline, zero internet or downloads needed).
- **Smooth 25x Zoom & Pan**: Mouse wheel zoom directly at the cursor (crucial for finding small solder balls on 1920x1080 PCBs).
- **Quick-Click Mode (`2`)**: Simply click directly on a solder ball to instantly box it.
- **Box Drag Mode (`1`)**: Drag a custom rectangle.
- **✨ Auto-Detect**: Pre-suggests candidate solder ball locations.
- **Auto-Save**: Saves directly to `annotations/<image_name>.txt` in YOLO format.

#### Option 2: Native PyQt5 Desktop GUI Window
Run:
```bash
python scripts/desktop_annotator.py
```
*(or double-click `start_desktop_gui.bat`)*

- Native desktop window with pan, zoom, shortcuts (`A` prev, `D` next, `Z` undo, `S` save).

> **💡 Time-Saving Tip (Active Learning):**
> You only need to annotate **25–35 images** manually! Once trained, use `python scripts/auto_annotate.py` to auto-label the rest.

---

### Step 2: Split Dataset & Verify
Once you have annotations ready:

```bash
# 1. Split into 80% train / 20% validation
python scripts/split_dataset.py --val-ratio 0.2

# 2. Visually verify the annotations
python scripts/verify_annotations.py --limit 10
```
This saves verification images with overlay boxes into `verification_output/`.

---

### Step 3: Train the Detector

```bash
python scripts/train.py --epochs 50 --imgsz 640 --batch 4
```
For higher precision on small solder balls, you can increase `--imgsz 1080`.
The best model will automatically be saved to `models/best_solderball_detector.pt`.

---

### Step 4: Run Detection & Counting

#### Option A: Interactive Web UI (Streamlit)
```bash
streamlit run app.py
```
- Upload images or select from the 174 dataset images.
- Adjust confidence threshold and IoU sliders in real time.
- View real-time solder ball counts, defect status (PASS/FAIL), and download CSV reports.

#### Option B: Command Line (Batch Processing)
```bash
# Single image
python scripts/detect_and_count.py --source "dataset/solderball (1).jpg"

# Whole directory (first 10 images)
python scripts/detect_and_count.py --source dataset/ --limit 10
```
Outputs annotated images with the total count banner and generates `detection_summary.csv`.
