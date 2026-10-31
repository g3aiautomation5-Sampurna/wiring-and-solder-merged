"""
Live Video / Webcam Detection for Origin 3
Runs real-time YOLO detection using connected camera (or video stream)
Displays bounding box, confidence score, and center coordinates (cx, cy).
"""

import sys
import os
import cv2
from ultralytics import YOLO

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKSPACE_DIR, "runs", "origin3_model", "weights", "best.pt")


def run_live(camera_id=0, conf_threshold=0.5):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Trained model weights not found at: {MODEL_PATH}")
        return

    print("--> Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print(f"--> Connecting to Camera ID: {camera_id} (Press 'q' in window to quit)...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}.")
        print("If using an external camera or RTSP stream, pass the camera index or stream URL:")
        print("  python detect_live.py 1")
        print("  python detect_live.py rtsp://...")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                # Draw center crosshair
                cv2.drawMarker(
                    annotated_frame,
                    (cx, cy),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=15,
                    thickness=2,
                )
                cv2.putText(
                    annotated_frame,
                    f"Center: ({cx}, {cy})",
                    (int(x1), int(y2) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Origin 3 Real-time Detector (Press Q to quit)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        cam = int(arg) if arg.isdigit() else arg
    run_live(cam)
