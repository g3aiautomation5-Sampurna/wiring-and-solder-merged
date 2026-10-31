# Origin 3 Object Detection Pipeline

This directory contains everything you need to annotate, prepare, train, and test an object detection model to detect **Origin 3** in your 67 images.

---

## 🚀 Quick Start Guide

### Step 1: Annotate Images Yourself
To label the single **Origin 3** object in each image:

- **Option A (Easiest)**: Double-click **`start_annotating.bat`** in this folder.
- **Option B (Terminal)**: Run:
  ```powershell
  python annotate.py
  ```

#### Annotator Shortcuts & Controls:
- **Left Click & Drag**: Draw bounding box around *Origin 3*.
- **Right Click** or **Delete / Backspace / C**: Clear current bounding box.
- **D** or **Right Arrow**: Next image (automatically saves current box).
- **A** or **Left Arrow**: Previous image.
- **W**: Jump to **Next Unlabeled** image.
- **S**: Save current annotation.
- **Jump to dropdown**: Jump directly to any of the 67 images.

> Annotations are automatically saved in YOLO format (`0 x_center y_center width height`) inside `dataset/labels/`.

---

### Step 2: Prepare YOLO Dataset
Once you have labeled your images (or a good subset to start testing):
```powershell
python prepare_dataset.py
```
This script will:
1. Verify how many images are annotated.
2. Split them into 80% Training and 20% Validation sets.
3. Organize them into `yolo_dataset/images/` and `yolo_dataset/labels/`.
4. Generate `yolo_dataset/data.yaml`.

---

### Step 3: Train YOLOv8 Model
Train a YOLOv8 nano model on your dataset:
```powershell
python train.py
```
- Trains for 50 epochs (automatically using GPU if available, or CPU).
- Produces training metrics, loss graphs, and the best model weights at:  
  `runs/origin3_model/weights/best.pt`

---

### Step 4: Run Detection (Inference)

#### A. Run on Images
Test your trained model on images:
```powershell
# Run on sample validation images
python detect.py

# Or test on a specific image
python detect.py dataset/20260922_114420_test_raw_orig3.jpg

# Or run on an entire folder
python detect.py dataset/
```
The predicted image with bounding box, confidence score, and center coordinates `(x, y)` will be saved in `predictions/`.

#### B. Run Real-time on Live Camera / Webcam

##### Option 1: Browser-Based Dual Camera Platform
Launch the web dashboard:
```powershell
run_dual_web.bat
```
1. Select **Cam 1 (Top Pose)** and **Cam 2 (Origin 3)** from the dropdowns.
2. Click **`▶ START INSPECTION`**:
   - Both cameras are automatically clicked simultaneously with a visual shutter flash.
   - Evaluates Top View (Wire Pose) and Side View (Origin 3).
   - Displays the **OVERALL: PASS / FAIL** banner and full breakdown.
3. Click **`🔄 Live Stream`** to resume live feeds for the next unit.
*(You can also turn on **Continuous Auto Mode** to auto-inspect every 1.5 seconds).*

##### Option 2: Native Desktop OpenCV Dual Camera Rig
Launch the desktop rig:
```powershell
start_dual_camera_rig.bat
# Or: python run_dual_camera_rig.py 0 1
```
- Displays both camera feeds live side-by-side.
- Press **`SPACE`** or **`S`** to automatically click both cameras and inspect!
- Press **`SPACE`** or **`R`** to return to live stream for the next unit.
