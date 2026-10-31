# Wire Harness Inspection POC: Keypoint / Pose Estimation

This POC replaces fragile fixed pixel ROIs and HSV color-mask centroids with a **deformable Keypoint / Pose Estimation model (YOLOv8-Pose)**.

---

## Why this is Fail-Proof against Wire & Part Shifts
1. **Dynamic Detection**: The neural network locates the wires and their keypoints wherever they appear in the camera frame — no hardcoded pixel boxes.
2. **Topological Rules (Origins & Terminals)**:
   - **Upper Middle (UM)**:
     - **Origin**: Red wire's origin must be on the **left** of Black wire's origin (`X_red < X_black`).
     - **Terminal/Socket**: Red wire's socket/terminal must be on the **right** of Black wire's socket/terminal (`X_red > X_black`).
   - **Below Left (BL)**:
     - **Origin**: Red wire's origin must be **below** Black wire's origin (`Y_red > Y_black`).
     - **Terminal/Socket**: Red wire's socket/terminal must be on the **left** of Black wire's socket/terminal (`X_red < X_black`).
   - **Below Right (BR)**:
     - **Origin**: Both wire origins must **exist** (both Red and Black origins present).
     - **Terminal/Socket**: Red wire's socket/terminal must be on the **left** of Black wire's socket/terminal (`X_red < X_black`).
   - **Position Invariance**: Because the evaluation measures relative topological ordering of Keypoint 0 (Origin) and Keypoint 2 (Terminal/Socket), part shifts, rotations, or slack do not produce false alarms.

---

## Directory Structure

```
pose_estimation_poc/
├── data.yaml                     # YOLOv8-Pose dataset configuration
├── annotate_wires.py             # Interactive GUI tool to click & label keypoints
├── bootstrap_starter_dataset.py  # Auto-generated initial 20 samples to test immediately
├── train_pose.py                 # Ultralytics YOLOv8-pose training script
├── inspect_pose.py               # Inspection & Pass/Fail rules engine
├── dataset/
│   ├── images/train & val/       # Training & validation images
│   └── labels/train & val/       # YOLO Pose label text files
└── test_samples/                 # Test images (Correct and Case 1, 2, 3 Failures)
```

---

## Step-by-Step Instructions

### Step 1: Annotate Additional Images (Optional / Ongoing)
An interactive labeling tool is included:
```bash
python pose_estimation_poc/annotate_wires.py
```
* **Controls in GUI:**
  * Press `[R]` to set active wire to **Red (Class 0)**.
  * Press `[B]` to set active wire to **Black (Class 1)**.
  * Click 3 points per wire in order:
    1. **Origin** (Upper coil connection)
    2. **Midpoint** (Grommet / center bend)
    3. **Terminal** (Lower socket connection)
  * Press `[SPACE]` or `[S]` to save and load the next image.
  * Press `[Z]` to undo last point or wire.
  * Press `[Q]` or `[ESC]` to exit.

> Note: 20 starter samples have already been populated using `bootstrap_starter_dataset.py` so you can test training right away!

---

### Step 2: Train the Pose Model
Run:
```bash
python pose_estimation_poc/train_pose.py
```
* Automatically uses your GPU if available, or CPU.
* Outputs the trained weights to:
  `pose_estimation_poc/runs/wire_pose_model/weights/best.pt`

---

### Step 3: Run the Inspection Rules Engine (CLI)
Run:
```bash
python inspect_pose.py
```
* Evaluates all images in `test_samples/` (including good samples and Case 1, 2, 3 failure modes).
* Runs relative origin and socket topological checks.
* Saves visual inspection overlays with PASS / FAIL badges into:
  `runs/inspections/`

---

### Step 4: Run the Browser-Based Inspection Dashboard (Recommended)
Launch the modern web UI by double-clicking **`run_web.bat`** or running:
```bash
python web_app.py --port 5000
```
This automatically opens **`http://localhost:5000`** in your browser.

* **Key Web Features**:
  * **Live Camera Feed**: Supports USB/Arducam webcams with device selection and HD streaming.
  * **Dual Inspection Mode**:
    * **Snap & Inspect**: Captures a snapshot of the live camera stream for immediate analysis.
    * **Live Auto-Inspect Toggle**: Continuously runs automated inspection on the live video stream.
  * **Image Upload & Drag-and-Drop**: Easily upload wire harness photos directly into the browser.
  * **1-Click Test Samples**: Instantly load and test reference samples (`Correct 1`, `Correct 2`, `Case 1 BL`, `Case 2 UM`, `Case 3 BR`).
  * **Neatly Arranged Results**:
    * **Overall Banner**: Bold PASS / FAIL verdict badge with latency and wire detection count.
    * **Component Breakdown Cards**: Individual visual cards for **Upper Middle**, **Below Left**, and **Below Right**.
    * **Sub-check Details**: Discrete origin and terminal checks with detected coordinates and pass/fail pills.
    * **Defects Alert Box**: Pinpoints exact wiring inversions or missing wires.
    * **Pose Overlay & Toggle**: View the annotated keypoint skeleton with `R-Orig`, `B-Orig`, `R-Term`, and `B-Term` markers or switch to the original photo.
  * **Annotation & Retraining Studio (Tab 2)**:
    * **Dynamic Class Management**: Add or remove custom wire classes/colors (e.g., `wire_blue`, `wire_yellow`, `wire_white`). Automatically syncs with `data.yaml`.
    * **In-Canvas Class Selection (Dropdown & Tick Box)**: Switch active wire classes directly on the canvas toolbar using either a **Dropdown menu** or **Tick boxes (radio chips)** while annotating (including in fullscreen mode), or using keyboard shortcuts (`R`, `B`, `1`-`9`).
    * **3-Point Keypoint Annotation**: Identical pose estimation labeling method (Click 1: Origin, Click 2: Midpoint, Click 3: Terminal).
    * **Fullscreen Annotation Mode (`⛶ Fullscreen`)**: Maximize the canvas to your full monitor display for pixel-precise keypoint clicking, with ESC hotkey support to exit.
    * **Delete Image & Labels (`🗑️ Delete Image`)**: Delete unwanted images and their corresponding label files directly from the image list or toolbar, automatically purging cache for clean training.
    * **Edit Existing Labels**: Load existing images from the dataset, view labeled wires, edit, delete, or add wires.
    * **In-Browser Retraining Engine**: Train or fine-tune YOLOv8-pose with customizable epochs & batch size.
    * **Live Training Telemetry**: Real-time progress bar, epoch counter, and streaming log console with automatic model hot-reloading upon completion.

---

### Step 5: Run the Desktop GUI (PyQt6 Alternative)
If you prefer a standalone desktop window without a browser:
```bash
python gui_app.py
```
or double-click **`run_gui.bat`**.

---

## How to Retrain with New Incoming Data
Whenever production operators log new samples or edge cases:
1. Place new images in your raw image folder.
2. Label them with `annotate_wires.py`.
3. Re-run `python pose_estimation_poc/train_pose.py`.
   - The weights will update automatically in `runs/wire_pose_model/weights/best.pt`.
   - No code changes to the inspection engine are required!
