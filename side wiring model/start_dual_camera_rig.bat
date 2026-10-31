@echo off
title Dual Camera Hardware Rig
cd /d "%~dp0"
echo ========================================================
echo Starting Dual Camera Inspection Rig (OpenCV Window)
echo Press [SPACE] or [S] to auto-click both cameras!
echo ========================================================
python run_dual_camera_rig.py 0 1
pause
