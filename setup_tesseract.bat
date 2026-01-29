@echo off
REM Tesseract OCR Configuration Script
REM Run this before starting the application if OCR is not working

echo Configuring Tesseract OCR for Windows...

REM Add Tesseract to PATH for this session
set PATH=%PATH%;C:\Program Files\Tesseract-OCR

REM Verify installation
tesseract --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Tesseract OCR not found!
    echo Please install Tesseract OCR:
    echo   Option 1: winget install --id UB-Mannheim.TesseractOCR
    echo   Option 2: Download from https://github.com/UB-Mannheim/tesseract/wiki
    pause
    exit /b 1
)

echo.
echo ✅ Tesseract OCR is configured and ready!
echo.
