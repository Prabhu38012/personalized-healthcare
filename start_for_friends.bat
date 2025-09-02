@echo off
echo ========================================
echo  Healthcare App - Quick Start for Friends
echo ========================================
echo.
echo This script will help you get started after pulling the latest code.
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo Please make sure Python is installed and added to PATH
        pause
        exit /b 1
    )
)

echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [SETUP] Installing/updating dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Environment setup complete!
echo.
echo Choose what to do next:
echo 1. Run diagnostic check
echo 2. Start the application automatically
echo 3. Manual startup instructions
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo [DIAGNOSTIC] Running authentication diagnostics...
    python debug_auth.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo [AUTO-START] Starting application automatically...
    python fix_login_for_friends.py
) else if "%choice%"=="3" (
    echo.
    echo [MANUAL] Manual startup instructions:
    echo.
    echo 1. Open TWO command prompt windows
    echo 2. In both windows, navigate to this directory and run: venv\Scripts\activate
    echo 3. In first window run: python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
    echo 4. In second window run: python -m streamlit run frontend/app.py --server.port 8501
    echo 5. Open browser to: http://localhost:8501
    echo.
    echo Test accounts:
    echo - Admin: admin@healthcare.com / Admin123!
    echo - Doctor: doctor@healthcare.com / Doctor123!
    echo - Patient: patient@healthcare.com / Patient123!
    echo.
    pause
) else (
    echo Exiting...
)

echo.
echo For more detailed instructions, see QUICK_START_FOR_FRIENDS.md
pause