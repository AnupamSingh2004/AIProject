@echo off
echo ============================================
echo Fashion AI - Complete Setup with GPU Support
echo ============================================
echo.

echo [1/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/5] Installing TensorFlow (with GPU support)...
pip install tensorflow

echo.
echo [3/5] Installing Computer Vision libraries...
pip install opencv-python pillow numpy

echo.
echo [4/5] Installing ML libraries...
pip install scikit-learn scikit-image matplotlib pandas seaborn

echo.
echo [5/5] Installing Web API libraries...
pip install fastapi uvicorn python-multipart

echo.
echo ============================================
echo Testing GPU availability...
echo ============================================
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs found: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus] if gpus else print('  No GPU detected - will use CPU')"

echo.
echo ============================================
echo Installation complete!
echo ============================================
echo.
echo To start the backend:
echo   cd backend
echo   python main.py
echo.
echo To start the frontend:
echo   cd fashion-recommender
echo   npm run dev
echo.
pause
