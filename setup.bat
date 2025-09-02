@echo off
echo 🏥 Starting Personalized Healthcare System...
echo.

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
echo    1. Backend:  cd backend ^&^& python app.py
echo    2. Frontend: cd frontend ^&^& streamlit run app.py
echo.
echo 🌐 Then open: http://localhost:8501
echo.
pause