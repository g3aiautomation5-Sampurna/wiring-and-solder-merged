@echo off
title Wire Harness Pose Inspection - Browser Dashboard
cd /d "%~dp0"
echo ===================================================
echo  Starting Wire Harness Pose Inspection Web Server
echo  Opening browser at http://localhost:5000 ...
echo ===================================================
python web_app.py --port 5000
pause
