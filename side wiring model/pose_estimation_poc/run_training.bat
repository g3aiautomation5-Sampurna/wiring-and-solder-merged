@echo off
title YOLOv8-Pose Training
cd /d "%~dp0"
echo =======================================================
echo Starting Wire Pose Model Training...
echo =======================================================
python train_pose.py
echo.
echo Training process finished.
pause
