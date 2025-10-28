@echo off
REM Setup script for Fashion AI Backend with GPU support
echo ========================================
echo Fashion AI Backend - GPU Setup
echo ========================================
echo.

REM Check if CUDA is installed
echo Checking for CUDA installation...
where nvcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] CUDA compiler found
    nvcc --version
) else (
    echo [WARNING] CUDA not found in PATH
    echo.
    echo To use GPU acceleration:
    echo 1. Download CUDA 11.8 or 12.x from: https://developer.nvidia.com/cuda-downloads
    echo 2. Download cuDNN from: https://developer.nvidia.com/cudnn
    echo 3. Add CUDA to your PATH
    echo.
)

REM Check if nvidia-smi is available
echo.
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
) else (
    echo [WARNING] nvidia-smi not found
    echo Make sure NVIDIA drivers are installed
)

echo.
echo ========================================
echo Installing Python Dependencies
echo ========================================
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install TensorFlow with GPU support
echo.
echo Installing TensorFlow with GPU support...
pip install tensorflow[and-cuda]==2.15.0

REM Install other requirements
echo.
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Testing GPU Configuration
echo ========================================
echo.

REM Test GPU setup
python gpu_config.py

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the backend server:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Run server: python main.py
echo.
pause
