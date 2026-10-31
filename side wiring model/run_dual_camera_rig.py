"""
Dual Camera Hardware Rig (OpenCV Native)
Camera 0 -> Top Camera (Wire Harness Pose)
Camera 1 -> Side Camera (Origin 3 Detection)
Press [SPACE] or [S] to automatically click both cameras and run dual inspection!
"""

import sys
import os
import cv2
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_DIR = os.path.join(WORKSPACE_DIR, "pose_estimation_poc")
if POSE_DIR not in sys.path:
    sys.path.insert(0, POSE_DIR)

from dual_inspection import inspect_dual_system


def run_rig(cam_id_top=0, cam_id_orig3=1):
    print("\n=======================================================")
    print(" DUAL CAMERA RIG (START ON DEMAND)")
    print("=======================================================")
    print(f" Connecting to Top Camera  (ID: {cam_id_top})...")
    print(f" Connecting to Side Camera (ID: {cam_id_orig3})...")
    print(" Controls:")
    print("   [SPACE] or [S] : START (Clicks both cameras & inspects)")
    print("   [R]            : Resume / Live Stream")
    print("   [Q] or [ESC]   : Exit")
    print("=======================================================\n")

    cap_top = cv2.VideoCapture(cam_id_top)
    cap_orig3 = cv2.VideoCapture(cam_id_orig3)

    top_ok = cap_top.isOpened()
    orig3_ok = cap_orig3.isOpened()

    if not top_ok and not orig3_ok:
        print("[ERROR] Neither camera could be opened.")
        print("Please check your USB camera connections.")
        return

    # If only one camera exists on the PC, duplicate stream for demonstration
    single_cam_mode = False
    if top_ok and not orig3_ok:
        print("[INFO] Only 1 camera detected. Running in single-camera mirror test mode.")
        single_cam_mode = True

    cv2.namedWindow("Dual Camera Inspection Station", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dual Camera Inspection Station", 1280, 520)

    is_showing_result = False
    result_display = None
    is_fullscreen = False

    while True:
        if not is_showing_result:
            # Live Feed Mode
            ret_top, frame_top = cap_top.read() if top_ok else (False, None)
            if single_cam_mode:
                ret_orig3, frame_orig3 = ret_top, frame_top
            else:
                ret_orig3, frame_orig3 = cap_orig3.read() if orig3_ok else (False, None)

            # Fallback if frame failed
            if frame_top is None:
                frame_top = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame_top, "Top Camera Not Connected", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if frame_orig3 is None:
                frame_orig3 = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame_orig3, "Side Camera Not Connected", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Resize to uniform display height (480)
            h = 480
            w_top = int(frame_top.shape[1] * (h / frame_top.shape[0]))
            w_orig3 = int(frame_orig3.shape[1] * (h / frame_orig3.shape[0]))
            d_top = cv2.resize(frame_top, (w_top, h))
            d_orig3 = cv2.resize(frame_orig3, (w_orig3, h))

            # Add labels
            cv2.rectangle(d_top, (10, 10), (320, 50), (15, 23, 42), -1)
            cv2.putText(d_top, "CAM 1: Top View (Pose)", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.rectangle(d_orig3, (10, 10), (320, 50), (15, 23, 42), -1)
            cv2.putText(d_orig3, "CAM 2: Side View (Origin 3)", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Combined canvas
            canvas = np.hstack([d_top, d_orig3])

            # Bottom Action Banner
            banner = np.zeros((70, canvas.shape[1], 3), dtype=np.uint8)
            cv2.putText(banner, ">>> PRESS [SPACE] OR [S] TO START DUAL INSPECTION | [F] FULLSCREEN <<<", (canvas.shape[1]//2 - 430, 42), cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 255, 128), 2)
            display = np.vstack([canvas, banner])

            cv2.imshow("Dual Camera Inspection Station", display)
            key = cv2.waitKey(15) & 0xFF

            if key in [ord('f'), ord('F')]:
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty("Dual Camera Inspection Station", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)

            elif key in [ord(' '), ord('s'), ord('S')]:
                # FLASH EFFECT
                flash = np.ones_like(display) * 255
                cv2.imshow("Dual Camera Inspection Station", flash)
                cv2.waitKey(60)

                print("\n[ACTION] START Triggered! Auto-clicking both cameras...")
                # Run inspection on captured raw frames
                res = inspect_dual_system(frame_top, frame_orig3)
                
                # Render combined result screen
                p_annotated = res["pose_result"].get("annotated_img", frame_top)
                o_annotated = res["origin3_result"].get("annotated_img", frame_orig3)

                r_top = cv2.resize(p_annotated, (w_top, h))
                r_orig3 = cv2.resize(o_annotated, (w_orig3, h))
                res_canvas = np.hstack([r_top, r_orig3])

                # Large Verdict Header
                overall_pass = res["overall_pass"]
                vh_color = (0, 180, 0) if overall_pass else (0, 0, 220)
                v_header = np.zeros((90, res_canvas.shape[1], 3), dtype=np.uint8)
                v_header[:] = vh_color

                v_text = "OVERALL RESULT: PASS (All Checks OK)" if overall_pass else "OVERALL RESULT: FAIL"
                cv2.putText(v_header, v_text, (30, 45), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
                
                sub_text = f"Top Pose: {'PASS' if res['pose_pass'] else 'FAIL'}  |  Origin 3: {'PASS' if res['orig3_pass'] else 'FAIL'}  |  Latency: {res['total_latency_ms']} ms"
                cv2.putText(v_header, sub_text, (32, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

                # Bottom control
                res_footer = np.zeros((50, res_canvas.shape[1], 3), dtype=np.uint8)
                cv2.putText(res_footer, "Press [SPACE] / [R] for Next Unit  |  [F] Fullscreen  |  [Q] Exit", (res_canvas.shape[1]//2 - 280, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

                result_display = np.vstack([v_header, res_canvas, res_footer])
                is_showing_result = True

            elif key in [ord('q'), ord('Q'), 27]:
                break

        else:
            # Showing Inspected Result Mode
            cv2.imshow("Dual Camera Inspection Station", result_display)
            key = cv2.waitKey(30) & 0xFF
            if key in [ord('f'), ord('F')]:
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty("Dual Camera Inspection Station", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            elif key in [ord(' '), ord('r'), ord('R'), ord('s'), ord('S')]:
                is_showing_result = False
            elif key in [ord('q'), ord('Q'), 27]:
                break

    if top_ok: cap_top.release()
    if orig3_ok and not single_cam_mode: cap_orig3.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    c1 = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    c2 = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
    run_rig(c1, c2)
