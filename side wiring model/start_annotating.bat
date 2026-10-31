@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting Origin 3 Object Detection Annotation Tool...
echo ========================================================
python annotate.py
pause
