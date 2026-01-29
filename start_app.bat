@echo off
REM Healthcare Application Launcher
REM Always uses the correct virtual environment Python

echo.
echo ============================================================
echo    PERSONALIZED HEALTHCARE SYSTEM - LAUNCHER
echo ============================================================
echo.

REM Configure Tesseract OCR path for document/prescription processing
set PATH=%PATH%;C:\Program Files\Tesseract-OCR

echo Using Virtual Environment Python...
echo.

cd /d D:\personalized-healthcare
D:\personalized-healthcare\venv\Scripts\python.exe start.py

pause
