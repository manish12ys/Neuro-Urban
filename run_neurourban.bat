@echo off
REM NeuroUrban Windows Launcher Script
REM This script helps run NeuroUrban on Windows with proper Unicode support

echo ========================================
echo NeuroUrban: AI-Powered City Planning
echo ========================================

REM Set UTF-8 encoding for console
chcp 65001 >nul 2>&1

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Checking installation...

REM Test the Unicode fix
echo Testing Unicode support...
python test_unicode_fix.py
if errorlevel 1 (
    echo ERROR: Unicode test failed
    pause
    exit /b 1
)

echo.
echo Unicode test passed! Starting NeuroUrban...
echo.

REM Ask user which mode to run
echo Choose how to run NeuroUrban:
echo 1. CLI Mode (Command Line Interface)
echo 2. Web Dashboard (Streamlit)
echo 3. Test Installation
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo Starting CLI mode...
    python main.py
) else if "%choice%"=="2" (
    echo Starting web dashboard...
    echo Open your browser to: http://localhost:8501
    streamlit run streamlit_app.py
) else if "%choice%"=="3" (
    echo Running installation test...
    python test_installation.py
) else (
    echo Invalid choice. Starting CLI mode by default...
    python main.py
)

echo.
echo NeuroUrban session ended.
pause
