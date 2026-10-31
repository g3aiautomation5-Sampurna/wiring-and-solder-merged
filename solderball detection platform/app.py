"""
Solder Ball Inspection & Counting Dashboard
Interactive Streamlit Application
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Solder Ball Detection & Counting Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 5px solid #ff4b4b;
    }
    .metric-card-pass {
        border-left: 5px solid #00c853;
    }
    .big-stat {
        font-size: 32px;
        font-weight: 700;
        margin: 5px 0;
    }
    .stat-label {
        color: #666;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detection_model(model_path):
    if os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            return YOLO(model_path), "YOLO (Trained)"
        except Exception as e:
            return None, f"Error: {e}"
    return None, "Computer Vision (Fallback)"

def run_inference(model, image_bgr, conf_thresh, iou_thresh):
    if model is not None:
        results = model.predict(image_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': conf
                })
        return detections
    else:
        # Fallback CV circle detector
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=15, 
            param1=50, 
            param2=28, 
            minRadius=3, 
            maxRadius=25
        )
        detections = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                x, y, r = c[0], c[1], c[2]
                detections.append({
                    'box': (int(x - r), int(y - r), int(x + r), int(y + r)),
                    'conf': 0.85
                })
        return detections

def draw_boxes_on_image(image_bgr, detections):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    total_count = len(detections)

    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['box']
        conf = det['conf']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Highlight box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, ((x1+x2)//2, (y1+y2)//2), 2, (0, 0, 255), -1)

        label = f"#{idx} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lbl_y1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, lbl_y1), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Header banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    status_str = f"SOLDER BALL COUNT: {total_count}"
    status_col = (0, 0, 255) if total_count > 0 else (0, 255, 0)
    cv2.putText(annotated, status_str, (18, 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, status_col, 2)

    return annotated

# Sidebar
st.sidebar.title("🔍 Solder Ball Inspector")
st.sidebar.markdown("Automated Optical Inspection (AOI) System")

# Model configuration
model_path = st.sidebar.text_input("Model Weights Path", "models/best_solderball_detector.pt")
model, model_name = load_detection_model(model_path)
st.sidebar.caption(f"**Active Engine:** {model_name}")

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
iou_thresh = st.sidebar.slider("IoU NMS Threshold", 0.1, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Navigation")
nav = st.sidebar.radio("View", ["Inspect & Count", "Dataset & Annotation Progress", "Annotation Guide"])

# Load dataset images
dataset_images = sorted(glob.glob("dataset/*.jpg"))

if nav == "Inspect & Count":
    st.title("🔬 Solder Ball Detection & Counting")
    st.markdown("Analyze PCB images to detect, localize, and count all present solder balls.")

    # Image source selector
    col_src1, col_src2 = st.columns([1, 2])
    with col_src1:
        source_mode = st.radio("Image Source", ["Select from Dataset", "Upload Image"])

    img_bgr = None
    img_name = ""

    if source_mode == "Select from Dataset":
        with col_src2:
            if dataset_images:
                selected_file = st.selectbox("Choose Image", dataset_images, index=0)
                img_name = os.path.basename(selected_file)
                img_bgr = cv2.imread(selected_file)

                # Option to delete image
                with st.expander("🗑️ Delete this image from dataset"):
                    st.caption(f"Permanently delete `{img_name}` and its annotation file.")
                    confirm_del = st.checkbox(f"I confirm I want to delete {img_name}", key=f"del_{img_name}")
                    if st.button("Delete Image Now", type="primary", disabled=not confirm_del):
                        bnoext = os.path.splitext(img_name)[0]
                        ann_p = os.path.join("annotations", f"{bnoext}.txt")
                        if os.path.exists(selected_file):
                            os.remove(selected_file)
                        if os.path.exists(ann_p):
                            os.remove(ann_p)
                        st.success(f"Deleted {img_name} successfully!")
                        st.rerun()
            else:
                st.warning("No images found in dataset/ folder.")
    else:
        with col_src2:
            uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                img_name = uploaded_file.name
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is not None:
        # Run detection
        detections = run_inference(model, img_bgr, conf_thresh, iou_thresh)
        count = len(detections)
        annotated_bgr = draw_boxes_on_image(img_bgr, detections)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Metric cards
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card metric-card-pass">
                <div class="stat-label">Solder Balls Found</div>
                <div class="big-stat" style="color: #0088ff;">{count}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card metric-card-pass">
                <div class="stat-label">Image Resolution</div>
                <div class="big-stat" style="font-size: 22px;">{img_bgr.shape[1]} x {img_bgr.shape[0]}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card metric-card-pass">
                <div class="stat-label">Detector Engine</div>
                <div class="big-stat" style="font-size: 18px;">{model_name}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Image Display Tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Annotated View", "🖼️ Original Image", "📋 Detection Table"])

        with tab1:
            st.image(annotated_rgb, caption=f"Detection Results: {count} solder ball(s) localized", use_container_width=True)

        with tab2:
            st.image(original_rgb, caption=f"Original PCB: {img_name}", use_container_width=True)

        with tab3:
            if detections:
                table_data = []
                for idx, det in enumerate(detections, 1):
                    x1, y1, x2, y2 = det['box']
                    table_data.append({
                        "Ball #": idx,
                        "Confidence": f"{det['conf'] * 100:.1f}%",
                        "X1": x1,
                        "Y1": y1,
                        "X2": x2,
                        "Y2": y2,
                        "Width (px)": x2 - x1,
                        "Height (px)": y2 - y1,
                    })
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
                
                # Download CSV
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Detection Report (CSV)", csv_bytes, f"report_{img_name}.csv", "text/csv")
            else:
                st.info("No solder balls detected on this board.")

elif nav == "Dataset & Annotation Progress":
    st.title("📊 Dataset & Annotation Progress")
    
    total_imgs = len(dataset_images)
    
    # Check annotations
    ann_files = glob.glob("annotations/*.txt") + glob.glob("dataset/*.txt") + glob.glob("dataset/*.json")
    annotated_bases = set([os.path.splitext(os.path.basename(f))[0] for f in ann_files])
    annotated_count = len(annotated_bases)
    remaining_count = max(0, total_imgs - annotated_count)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Images", total_imgs)
    c2.metric("Annotated Images", annotated_count)
    c3.metric("Remaining Images", remaining_count)

    progress_val = (annotated_count / total_imgs) if total_imgs > 0 else 0
    st.progress(progress_val)
    st.write(f"**Progress: {progress_val * 100:.1f}% Complete**")

    st.markdown("---")
    st.subheader("Training/Validation Split Status")
    train_count = len(glob.glob("data/images/train/*"))
    val_count = len(glob.glob("data/images/val/*"))
    st.write(f"- Images in `data/images/train/`: **{train_count}**")
    st.write(f"- Images in `data/images/val/`: **{val_count}**")

    if annotated_count > 0:
        st.success(f"You have {annotated_count} annotated images ready! Run `python scripts/split_dataset.py` to prepare them for training.")
    else:
        st.info("No annotations found yet. Follow the 'Annotation Guide' tab to start annotating.")

elif nav == "Annotation Guide":
    st.title("📘 Annotation Guide & Next Steps")
    st.markdown("""
    ### Best Ways to Annotate Solder Balls

    #### Method 1: Roboflow (Fastest & Highly Recommended)
    1. Go to [Roboflow.com](https://roboflow.com) (free account).
    2. Create a new project: **Object Detection**, project name: `Solder Ball Detection`.
    3. Upload the images from your `dataset/` folder.
    4. Draw bounding boxes around each solder ball with label `solder_ball`.
    5. Click **Generate Version** -> **Export Dataset** -> Format: **YOLOv8**.
    6. Extract the downloaded zip into the `data/` folder in this workspace!

    #### Method 2: Offline Desktop Tool (LabelImg / Labelme)
    - You can use LabelImg or Labelme offline to save `.txt` or `.json` files.
    - Save the annotation files with the same name as the image (e.g. `solderball (1).txt`).
    - Our `scripts/split_dataset.py` handles auto-conversion and dataset splitting!

    #### 💡 Pro-Tip: Active Learning (Save 80% Time!)
    You do NOT need to manually annotate all 174 images from scratch!
    1. Annotate just **25 to 35 images** first.
    2. Run `python scripts/split_dataset.py`
    3. Run `python scripts/train.py --epochs 40`
    4. Run `python scripts/auto_annotate.py` to automatically predict boxes on the remaining images!
    5. Review and correct the predicted boxes.
    """)
