"""
Dual Camera / Dual Image Inspection CLI
Inspects:
  1. Top Camera Image -> Wire Harness Pose Estimation (UM, BL, BR)
  2. Side Camera Image -> Origin 3 Object Detection
Prints individual breakdowns and OVERALL SYSTEM VERDICT.
"""

import os
import sys

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_DIR = os.path.join(WORKSPACE_DIR, "pose_estimation_poc")
if POSE_DIR not in sys.path:
    sys.path.insert(0, POSE_DIR)

from dual_inspection import inspect_dual_system


def main():
    top_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(POSE_DIR, "test_samples", "correctwiring (1).jpg")
    orig3_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(POSE_DIR, "test_samples", "origin3", "orig3_pass.jpg")

    print(f"\n=======================================================")
    print(f" DUAL INSPECTION PIPELINE")
    print(f"=======================================================")
    print(f" Top Camera (Pose)  : {top_path}")
    print(f" Side Camera (Orig3): {orig3_path}")
    print(f"-------------------------------------------------------")

    res = inspect_dual_system(top_path, orig3_path)

    # 1. Pose Details
    p_res = res["pose_result"]
    print(f"\n[1] TOP VIEW (WIRE HARNESS POSE ESTIMATION):")
    print(f"    Status     : {'PASS' if res['pose_pass'] else 'FAIL'}")
    print(f"    Wires      : {p_res.get('total_wires', 0)} detected")
    print(f"    Latency    : {p_res.get('latency_ms', 0.0):.1f} ms")
    for comp_id, comp_name in [("UM", "Upper Middle"), ("BL", "Below Left"), ("BR", "Below Right")]:
        reg = p_res.get("regions", {}).get(comp_id, {})
        status = "PASS" if reg.get("pass") else "FAIL"
        print(f"    - {comp_name:<16}: [{status}] {reg.get('msg', '')}")

    # 2. Origin 3 Details
    o_res = res["origin3_result"]
    print(f"\n[2] SIDE VIEW (ORIGIN 3 OBJECT DETECTION):")
    print(f"    Status     : {'PASS' if res['orig3_pass'] else 'FAIL'}")
    print(f"    Confidence : {o_res.get('confidence', 0.0)*100:.1f}%")
    print(f"    Center (xy): {o_res.get('center')}")
    print(f"    BBox       : {o_res.get('bbox')}")
    print(f"    Detail     : {o_res.get('msg')}")

    # 3. Overall Verdict
    print(f"\n=======================================================")
    if res["overall_pass"]:
        print(f"  >>> OVERALL RESULT: PASS <<<")
        print(f"  Both Wire Harness and Origin 3 verified successfully!")
    else:
        print(f"  >>> OVERALL RESULT: FAIL <<<")
        print(f"  Failures detected:")
        for r in res["reasons_failed"]:
            print(f"   * {r}")
    print(f"  Total System Latency: {res['total_latency_ms']} ms")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
