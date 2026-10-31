@echo off
title Vision Quality Inspection Suite (Wiring & Soldering)
cd /d "%~dp0"
echo ======================================================================
echo           UNIFIED VISION QUALITY INSPECTION PLATFORM
echo     Wiring Pose Estimation & Solder Ball Detection in One App
echo ======================================================================
echo.
echo Launching unified platform in your default browser...
echo Local URL: http://localhost:5000
echo.
echo Tips:
echo   - Click 'Wiring Inspection' or press Alt+1 for Wire Harness Verification
echo   - Click 'Soldering Inspection' or press Alt+2 for Solder Ball Detection
echo.
python merged_app.py --port 5000
pause
