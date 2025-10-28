# 🚀 AI Fashion Recommender - Step-by-Step Running Guide

## ✅ **INSTALLATION COMPLETE!**

All necessary packages are now installed:
- ✅ TensorFlow 2.20.0
- ✅ Keras 3.12.0
- ✅ NVIDIA cuDNN (GPU libraries)
- ✅ NVIDIA cuBLAS (GPU libraries)
- ✅ FastAPI, Uvicorn, Pillow, NumPy

---

## 🎯 **STEP-BY-STEP COMMANDS TO RUN THE PROJECT**

### **Option 1: Automated Start (EASIEST)**

#### Step 1: Open PowerShell in project folder
```powershell
cd C:\Users\Prachi\Desktop\qq\AIProject
```

#### Step 2: Run the startup script
```powershell
.\start_all.bat
```

**That's it!** The script will:
- Start PostgreSQL database
- Start Backend API (port 8000)
- Start Next.js frontend (port 3000)
- Open your browser automatically

---

### **Option 2: Manual Start (Step-by-Step)**

Open **3 separate PowerShell terminals** and run these commands:

#### **Terminal 1 - Database**
```powershell
cd C:\Users\Prachi\Desktop\qq\AIProject
docker-compose up -d
```

**Wait 5 seconds**, then verify:
```powershell
docker ps
```
You should see `fashion_recommender_db` running.

---

#### **Terminal 2 - Backend API**
```powershell
cd C:\Users\Prachi\Desktop\qq\AIProject\backend
python start_backend.py
```

**You'll see:**
```
👕 Fashion AI Backend Server
Starting server at http://localhost:8000
API docs at http://localhost:8000/docs

🚀 Loading AI Models...
✅ Loaded successfully!
```

**Keep this terminal open!**

---

#### **Terminal 3 - Frontend**
```powershell
cd C:\Users\Prachi\Desktop\qq\AIProject\fashion-recommender
npm run dev
```

**You'll see:**
```
  ▲ Next.js 15.5.4
  - Local:        http://localhost:3000
  - Turbopack (beta) enabled
```

**Keep this terminal open!**

---

## 🌐 **Access Your Application**

Once all services are running:

1. **Frontend (Main App)**: http://localhost:3000
2. **Backend API Docs**: http://localhost:8000/docs
3. **Database**: localhost:5432 (internal)

---

## 🧪 **Test the System**

### Quick Test Command:
```powershell
python quick_test.py
```

### Test AI Model:
```powershell
python verify_model_works.py
```

### Check GPU Status:
```powershell
python setup_gpu.py
```

---

## 📝 **DETAILED RUNNING INSTRUCTIONS**

### **Method A: Using start_all.bat (Recommended)**

1. **Navigate to project folder**
   ```powershell
   cd C:\Users\Prachi\Desktop\qq\AIProject
   ```

2. **Double-click `start_all.bat`** OR run:
   ```powershell
   .\start_all.bat
   ```

3. **Wait 15-20 seconds** for all services to start

4. **Browser opens automatically** to http://localhost:3000

5. **Start using the app!**
   - Click "Wardrobe" in navigation
   - Click "Upload New Item"
   - Select a clothing image
   - See AI classification!

---

### **Method B: Manual Start (For Debugging)**

#### **Step 1: Start Docker Desktop**
- Open Docker Desktop application
- Wait until it shows "Docker Desktop is running"
- Verify with: `docker ps`

#### **Step 2: Start Database**
```powershell
cd C:\Users\Prachi\Desktop\qq\AIProject
docker-compose up -d
```

**Verify database is running:**
```powershell
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Should show:
```
fashion_recommender_db    Up X seconds (healthy)
```

#### **Step 3: Start Backend (New Terminal)**
```powershell
cd C:\Users\Prachi\Desktop\qq\AIProject\backend
python start_backend.py
```

**Expected output:**
```
======================================================================
👕 Fashion AI Backend Server
======================================================================
Starting server at http://localhost:8000
API docs at http://localhost:8000/docs
======================================================================

INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.

======================================================================
🚀 Loading AI Models...
======================================================================

📦 Loading Clothing Classifier...
✅ Loaded successfully!
   Layers: 7
   Input: (None, 224, 224, 3)
   Output: (None, 6)

======================================================================
✅ Backend Ready!
======================================================================

INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test backend:**
```powershell
# In another terminal:
curl http://localhost:8000/api/test-model
```

#### **Step 4: Start Frontend (New Terminal)**
```powershell
cd C:\Users\Prachi\Desktop\qq\AIProject\fashion-recommender
npm run dev
```

**Expected output:**
```
  ▲ Next.js 15.5.4
  - Local:        http://localhost:3000
  - Environments: .env
  - Experiments (use at your own risk):
    · turbopack

 ✓ Starting...
 ✓ Ready in 2.3s
```

#### **Step 5: Open Browser**
Visit: **http://localhost:3000**

---

## 🛑 **STOP THE PROJECT**

### **Option 1: Quick Stop**
```powershell
.\stop_all.bat
```

### **Option 2: Manual Stop**

1. **Stop Backend**: Press `Ctrl+C` in backend terminal

2. **Stop Frontend**: Press `Ctrl+C` in frontend terminal

3. **Stop Database**:
   ```powershell
   docker stop fashion_recommender_db
   ```

---

## 🎮 **GPU USAGE**

### **Check if GPU is Working:**
```powershell
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

### **GPU Libraries Installed:**
- ✅ nvidia-cudnn-cu12 (Deep Neural Network library)
- ✅ nvidia-cublas-cu12 (Linear Algebra library)

### **Note on GPU:**
Your RTX 3050 has:
- ✅ Driver installed (v581.57)
- ✅ CUDA libraries installed (via pip)
- ⚠️ GPU may not be detected without full CUDA Toolkit

**Your project WORKS on CPU** - GPU is optional for speed boost.

**To enable GPU (optional):**
1. Download CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-downloads
2. Install and restart computer
3. Run `python setup_gpu.py` to verify

---

## 📊 **VERIFICATION CHECKLIST**

Run these commands to verify everything is working:

```powershell
# 1. Check Docker
docker ps
# Should show: fashion_recommender_db

# 2. Check Backend
curl http://localhost:8000/api/test-model
# Should return JSON with model info

# 3. Check Frontend
curl http://localhost:3000
# Should return HTML

# 4. Check Models
dir models\saved_models
# Should show two .keras files

# 5. Check Python packages
pip list | findstr "tensorflow"
# Should show: tensorflow 2.20.0

# 6. Full system test
python quick_test.py
```

---

## 🎯 **USAGE EXAMPLES**

### **1. Upload Clothing via Frontend**
1. Open http://localhost:3000
2. Click "Wardrobe" in navigation
3. Click "Upload New Item" button
4. Select an image file (JPG, PNG)
5. Add details (color, category)
6. Click Upload
7. AI analyzes image automatically!

### **2. Test API Directly**
```powershell
# Test model endpoint
curl http://localhost:8000/api/test-model

# Analyze an image
curl -X POST "http://localhost:8000/api/analyze-clothing" -F "file=@C:\path\to\image.jpg"
```

### **3. View API Documentation**
Open: http://localhost:8000/docs

This shows:
- All available endpoints
- Request/response formats
- Interactive testing interface

---

## 🐛 **TROUBLESHOOTING**

### **Issue: "Docker is not running"**
**Solution:**
1. Open Docker Desktop
2. Wait for it to fully start
3. Verify with: `docker ps`

### **Issue: "Port 8000 already in use"**
**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID <number> /F
```

### **Issue: "Port 3000 already in use"**
**Solution:**
```powershell
# Find and kill process
netstat -ano | findstr :3000
taskkill /PID <number> /F
```

### **Issue: "ModuleNotFoundError: No module named 'tensorflow'"**
**Solution:**
```powershell
pip install tensorflow keras
```

### **Issue: "Database connection failed"**
**Solution:**
```powershell
# Restart database
docker-compose down
docker-compose up -d

# Wait 5 seconds
docker ps
```

### **Issue: "Model file not found"**
**Solution:**
```powershell
# Verify models exist
dir models\saved_models

# Should show:
# clothing_classifier.keras (22.54 MB)
# outfit_compatibility_advanced.keras (19.66 MB)
```

---

## 📁 **PROJECT STRUCTURE REFERENCE**

```
AIProject/
│
├── start_all.bat              # 🚀 Start everything
├── stop_all.bat               # 🛑 Stop everything
├── quick_test.py              # ✓ System test
├── setup_gpu.py               # 🎮 GPU setup
├── verify_model_works.py      # 🧪 Model test
│
├── backend/
│   ├── start_backend.py       # Main backend server
│   └── requirements.txt
│
├── fashion-recommender/
│   ├── app/                   # Next.js pages
│   ├── components/            # React components
│   └── package.json
│
├── models/saved_models/
│   ├── clothing_classifier.keras
│   └── outfit_compatibility_advanced.keras
│
└── docker-compose.yml         # Database config
```

---

## 🔑 **KEY COMMANDS SUMMARY**

```powershell
# START PROJECT
.\start_all.bat                      # Automated start

# OR MANUAL:
docker-compose up -d                 # Start database
cd backend && python start_backend.py  # Start backend
cd fashion-recommender && npm run dev  # Start frontend

# STOP PROJECT
.\stop_all.bat                       # Automated stop

# TEST
python quick_test.py                 # Full system test
python verify_model_works.py         # Test AI model
curl http://localhost:8000/api/test-model  # Test backend
curl http://localhost:3000           # Test frontend

# MONITOR
docker ps                            # Check database
docker logs fashion_recommender_db   # Database logs
nvidia-smi                           # Check GPU (if available)

# TROUBLESHOOT
netstat -ano | findstr "3000 8000 5432"  # Check ports
docker-compose down                  # Reset database
docker-compose up -d                 # Restart database
```

---

## ✅ **YOU'RE ALL SET!**

### **To start your project NOW:**

```powershell
cd C:\Users\Prachi\Desktop\qq\AIProject
.\start_all.bat
```

**Then visit: http://localhost:3000**

---

## 📞 **QUICK HELP**

| Problem | Command |
|---------|---------|
| Check if running | `docker ps` |
| Test backend | `curl http://localhost:8000/api/test-model` |
| Test frontend | Open http://localhost:3000 |
| View logs | `docker logs fashion_recommender_db` |
| Restart database | `docker-compose restart` |
| Full reset | `docker-compose down && docker-compose up -d` |
| Check GPU | `nvidia-smi` |
| Test model | `python verify_model_works.py` |

---

**Your AI Fashion Recommender is ready to run! 🎉**

**Status:**
- ✅ All packages installed
- ✅ GPU libraries installed
- ✅ Models verified working
- ✅ Database configured
- ✅ Frontend & Backend ready

**Just run: `start_all.bat` and you're live!**
