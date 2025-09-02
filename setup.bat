@echo off
echo 🏥 Starting Personalized Healthcare System...
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH.
    echo Please install Python 3.8+ and add it to your PATH.
    pause
    exit /b 1
)
echo.

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ❌ Error: requirements.txt not found.
    echo Please run this script from the personalized-healthcare directory.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment. Please ensure Python is installed.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ✅ Setup complete!
echo.
echo 📋 To run the application:
echo    1. Backend:  cd backend ^&^& uvicorn app:app --reload --host 0.0.0.0 --port 8000
echo    2. Frontend: cd frontend ^&^& streamlit run app.py --server.port 8501
echo.
echo 🌐 Then open: http://localhost:8501
echo.
pause