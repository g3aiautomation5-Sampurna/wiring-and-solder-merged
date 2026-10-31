@echo off
title Wire Harness & Origin 3 Dual Inspection Platform
cd /d "%~dp0\pose_estimation_poc"
echo ================================================================
echo  Starting Dual Camera Wire Harness & Origin 3 Inspection Server
echo  Opening browser at http://localhost:5000 ...
echo ================================================================
python web_app.py --port 5000
pause
