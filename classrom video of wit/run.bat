@echo off
cd /d "%~dp0"
echo ================================================
echo  SmartClass AI — Auto Setup and Run
echo ================================================
echo.
echo  This will:
echo    1. Check Python dependencies
echo    2. Download the classroom video (~957 MB) if needed
echo    3. Launch the analyzer
echo.
python setup.py
pause
