@echo off
REM Startup script for Web STT Demo (Windows)
REM This script starts both the FastAPI backend and Streamlit frontend

echo Starting Web Speech-to-Text Demo...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed
    exit /b 1
)

echo Starting FastAPI backend on port 8000...
start /B python api.py

REM Wait for API to start
timeout /t 3 /nobreak >nul

echo Starting Streamlit frontend on port 8501...
start /B streamlit run app.py

echo.
echo ==========================================
echo Web STT Demo is running!
echo ==========================================
echo API: http://localhost:8000
echo Web Interface: http://localhost:8501
echo.
echo Press Ctrl+C to stop the services
echo.

REM Keep the window open
pause
