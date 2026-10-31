@echo off
title Wire Pose Keypoint Annotator
cd /d "%~dp0"
echo Starting Wire Pose Annotation Tool...
python annotate_wires.py
pause
