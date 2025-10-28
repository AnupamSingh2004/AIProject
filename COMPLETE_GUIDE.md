# AI Fashion Recommender - Complete System Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Start Everything
```bash
# Double-click this file or run:
start_all.bat
```

This will automatically:
- ✅ Start PostgreSQL database
- ✅ Start Backend API (with GPU acceleration if available)
- ✅ Start Next.js frontend
- ✅ Open http://localhost:3000 in your browser

### Step 2: Use the Application
1. **Upload Clothing Items**: Go to "Wardrobe" and upload images
2. **AI Analysis**: Images are automatically analyzed by your trained model
3. **View Results**: See clothing categories and recommendations

### Step 3: Stop Everything
```bash
# Double-click this file or run:
stop_all.bat
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│              Next.js Frontend (Port 3000)                   │
│  - Wardrobe Management  - Image Upload  - Recommendations  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ↓ API Calls
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND API                               │
│              FastAPI Server (Port 8000)                     │
│  - REST Endpoints  - Model Inference  - GPU Acceleration   │
└─────────┬──────────────────────────┬────────────────────────┘
          │                          │
          ↓ SQL Queries              ↓ Model Predictions
┌─────────────────────┐    ┌──────────────────────────────────┐
│    DATABASE         │    │      AI MODELS                   │
│  PostgreSQL         │    │  - Clothing Classifier (6 cats) │
│  (Port 5432)        │    │  - Outfit Compatibility          │
│  - Users            │    │  - GPU Accelerated (RTX 3050)   │
│  - Clothing Items   │    │  - TensorFlow + CUDA            │
│  - Wardrobes        │    └──────────────────────────────────┘
│  - Recommendations  │
└─────────────────────┘
```

---

## 🎮 GPU Configuration

### Current Status
Your system has:
- ✅ **GPU**: NVIDIA RTX 3050 Laptop
- ✅ **Driver**: v581.57 (CUDA 13.0 compatible)
- ⚙️ **CUDA**: Installing with TensorFlow

### GPU Performance
- **Without GPU (CPU)**: ~3 seconds per image
- **With GPU**: ~0.5 seconds per image (6x faster!)

### Verify GPU is Working
```bash
python setup_gpu.py
```

---

## 🔧 Manual Startup (Advanced)

If you want to start services individually:

### 1. Start Database
```bash
docker-compose up -d
# or
docker start fashion_recommender_db
```

### 2. Start Backend
```bash
cd backend
python start_backend.py
```
Visit: http://localhost:8000/docs for API documentation

### 3. Start Frontend
```bash
cd fashion-recommender
npm run dev
```
Visit: http://localhost:3000

---

## 📁 Project Structure

```
AIProject/
├── start_all.bat              # 🚀 One-click startup
├── stop_all.bat               # 🛑 One-click shutdown
├── setup_gpu.py               # GPU setup and verification
├── verify_model_works.py      # Test AI models directly
├── docker-compose.yml         # Database configuration
│
├── backend/
│   ├── start_backend.py       # FastAPI server with models
│   ├── main.py                # Original backend
│   └── requirements.txt       # Python dependencies
│
├── fashion-recommender/       # Next.js frontend
│   ├── app/                   # Pages and routes
│   ├── components/            # React components
│   ├── prisma/                # Database schema
│   └── package.json           # Node dependencies
│
└── models/
    └── saved_models/
        ├── clothing_classifier.keras          (22.54 MB)
        └── outfit_compatibility_advanced.keras (19.66 MB)
```

---

## 🧪 Testing

### Test Database Connection
```bash
docker exec -it fashion_recommender_db psql -U fashion_user -d fashion_recommender
```

### Test Backend API
```bash
# Health check
curl http://localhost:8000/api/test-model

# Test image analysis
curl -X POST "http://localhost:8000/api/analyze-clothing" -F "file=@image.jpg"
```

### Test Model Directly
```bash
python verify_model_works.py
```

### Test Frontend
Open http://localhost:3000 and upload an image in the Wardrobe section

---

## 🎯 API Endpoints

### Backend (http://localhost:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/test-model` | GET | Test model status |
| `/api/analyze-clothing` | POST | Analyze clothing image |
| `/api/analyze-skin-tone` | POST | Analyze skin tone |
| `/docs` | GET | Interactive API docs |

### Frontend API Routes (http://localhost:3000)

| Route | Method | Description |
|-------|--------|-------------|
| `/api/wardrobe/items` | GET | Get all wardrobe items |
| `/api/wardrobe/items` | POST | Upload new clothing |
| `/api/wardrobe/items/[id]` | GET | Get single item |
| `/api/wardrobe/items/[id]/image` | GET | Get item image |

---

## 🐛 Troubleshooting

### Database Won't Start
```bash
# Check if port 5432 is in use
netstat -ano | findstr :5432

# Force restart
docker-compose down
docker-compose up -d
```

### Backend Won't Start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Reinstall dependencies
cd backend
pip install -r requirements.txt
```

### Frontend Won't Start
```bash
# Check if port 3000 is in use
netstat -ano | findstr :3000

# Reinstall dependencies
cd fashion-recommender
npm install
```

### GPU Not Detected
```bash
# Run GPU setup
python setup_gpu.py

# Check NVIDIA driver
nvidia-smi

# Reinstall TensorFlow with GPU
pip install --upgrade tensorflow[and-cuda]
```

### Model Loading Errors
```bash
# Verify models exist
dir models\saved_models

# Test model loading
python verify_model_works.py
```

---

## 📦 Dependencies

### Backend Requirements
- Python 3.13
- TensorFlow 2.20+ (with CUDA support)
- FastAPI
- Uvicorn
- Pillow
- NumPy

### Frontend Requirements
- Node.js 18+
- Next.js 15.5
- React 19
- Prisma ORM
- TailwindCSS

### System Requirements
- Windows 10/11
- Docker Desktop
- NVIDIA GPU (optional, for acceleration)
- 8GB+ RAM
- 5GB+ disk space

---

## 🔐 Database Credentials

**Connection String:**
```
postgresql://fashion_user:fashion_password_2024@localhost:5432/fashion_recommender
```

**Individual Credentials:**
- Host: `localhost`
- Port: `5432`
- Database: `fashion_recommender`
- Username: `fashion_user`
- Password: `fashion_password_2024`

---

## 📈 Performance Monitoring

### Check Backend Status
```bash
curl http://localhost:8000/api/test-model
```

### Check Database Status
```bash
docker ps --filter name=fashion_recommender_db
```

### Monitor GPU Usage
```bash
nvidia-smi
# or for continuous monitoring:
nvidia-smi -l 1
```

---

## 🎓 Usage Examples

### Upload Clothing via API
```python
import requests

with open('tshirt.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/analyze-clothing',
        files={'file': f}
    )
print(response.json())
```

### Add Item to Wardrobe
```bash
curl -X POST http://localhost:3000/api/wardrobe/items \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "demo@example.com",
    "category": "tops",
    "color": "blue",
    "image": "data:image/jpeg;base64,..."
  }'
```

---

## 🚀 Production Deployment

### Environment Variables
Create `.env` file:
```env
DATABASE_URL="postgresql://user:pass@host:5432/db"
AI_BACKEND_URL="http://backend:8000"
NODE_ENV="production"
```

### Build Frontend
```bash
cd fashion-recommender
npm run build
npm start
```

### Run Backend with Gunicorn
```bash
cd backend
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 📞 Support

- **Issues**: Check troubleshooting section above
- **GPU Problems**: Run `python setup_gpu.py` for diagnostics
- **Database Issues**: Run `docker-compose logs` to view logs
- **Model Errors**: Run `python verify_model_works.py` to test

---

## ✅ Checklist

Before reporting issues, verify:
- [ ] Docker is running
- [ ] Database container is healthy
- [ ] Backend is running on port 8000
- [ ] Frontend is running on port 3000
- [ ] Models exist in `models/saved_models/`
- [ ] GPU is detected (optional)
- [ ] All dependencies are installed

---

**Last Updated**: October 28, 2025  
**Version**: 1.0.0  
**GPU Support**: Enabled (RTX 3050)
