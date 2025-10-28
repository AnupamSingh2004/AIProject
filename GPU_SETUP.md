# GPU Setup Guide for RTX 3050 Laptop

## Current Status
- ✅ Docker PostgreSQL database running
- ✅ Next.js frontend connected to database  
- ✅ Backend API structure created
- ⚠️  TensorFlow GPU setup in progress

## Your GPU Information
- **GPU**: NVIDIA RTX 3050 Laptop
- **CUDA Version**: 12.9
- **Memory**: 4GB GDDR6

## To Complete GPU Setup

### Step 1: Install TensorFlow with GPU Support
```powershell
cd c:\Users\Prachi\Desktop\qq\AIProject
pip install tensorflow
```

TensorFlow 2.20+ automatically detects and uses CUDA 12.x on Windows.

### Step 2: Install Other ML Dependencies
```powershell
pip install opencv-python pillow numpy scikit-learn matplotlib pandas
pip install fastapi uvicorn python-multipart
pip install colorthief scikit-image
```

### Step 3: Test GPU Access
```powershell
python gpu_config.py
```

This will show if TensorFlow can see your RTX GPU.

### Step 4: Start the Backend API
```powershell
cd backend
python main.py
```

The backend will:
- Automatically detect your RTX GPU
- Load trained models (clothing_classifier.keras, outfit_compatibility_advanced.keras)
- Enable GPU memory growth (prevents OOM errors)
- Serve AI endpoints at http://localhost:8000

## Model Files Location
Your trained models are in:
```
models/saved_models/
├── clothing_classifier.keras
├── outfit_compatibility_advanced.keras
├── history.json
└── compatibility_advanced_history.json
```

## GPU Optimization Features

The backend is configured to:

1. **Auto-detect GPU**: Automatically finds and uses your RTX 3050
2. **Memory Growth**: Allocates GPU memory as needed (prevents crashes)
3. **Mixed Precision**: Uses FP16 for faster inference on RTX cards
4. **Batch Processing**: Optimizes for RTX architecture

## Alternative: Use Docker with GPU

If you prefer Docker with GPU support:

```powershell
docker-compose up -d ai_backend
```

But you'll need NVIDIA Container Toolkit installed first.

## Current Flow

### 1. Frontend (Next.js) - Port 3000
- Upload clothing images
- View wardrobe
- Get recommendations

### 2. Database (PostgreSQL) - Port 5432
- Stores clothing items as BYTEA
- User profiles
- Outfit recommendations

### 3. Backend API (FastAPI) - Port 8000
```
/api/analyze-clothing      - Classify clothing type & colors
/api/analyze-skin-tone     - Detect skin tone from photo  
/api/recommend-outfits     - Generate outfit combinations
```

### 4. AI Models (Running on RTX GPU)
- Clothing Classifier: Identifies clothing types
- Outfit Compatibility: Scores outfit combinations
- Color Analyzer: Extracts dominant colors
- Recommendation Engine: Combines all for suggestions

## Quick Start Command

Once dependencies are installed:

```powershell
# Terminal 1: PostgreSQL (already running)
docker ps

# Terminal 2: Backend API
cd backend
python main.py

# Terminal 3: Frontend
cd fashion-recommender
npm run dev
```

Then visit: http://localhost:3000/wardrobe

## Troubleshooting

### If GPU not detected:
1. Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
2. Restart computer after driver update
3. Run `nvidia-smi` to verify GPU is visible

### If models don't load:
Check that .keras files exist in `models/saved_models/`

### If Python 3.13 causes issues:
Some packages (like mediapipe) don't support Python 3.13 yet.
Consider using Python 3.11 or 3.12 if needed.

## Performance on RTX 3050

Expected inference times:
- Clothing classification: ~50-100ms per image
- Outfit compatibility: ~30-50ms per combination
- Skin tone analysis: ~200-300ms (uses face detection)

With GPU, these will be 3-5x faster than CPU.

## Next Steps

1. Complete TensorFlow installation
2. Test GPU detection
3. Start backend API
4. Upload test images through frontend
5. See AI-powered recommendations!
