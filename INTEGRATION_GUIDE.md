# 🎨 Fashion Recommender - Full Stack Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (NextJS)                 │
│  - Upload photos, manage wardrobe, view recommendations    │
│  - Port: 3000                                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              API Routes (NextJS API)                        │
│  - /api/skin-analysis - Analyze skin tone                  │
│  - /api/wardrobe/items - Manage clothing items             │
│  - /api/recommendations - Get outfit suggestions           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├────────────┬───────────────┐
                       ▼            ▼               ▼
┌────────────────┐  ┌──────────────────┐  ┌────────────────┐
│  PostgreSQL DB │  │  AI Backend API  │  │  File Storage  │
│  (Docker)      │  │  (FastAPI)       │  │  (Database)    │
│  Port: 5432    │  │  Port: 8000      │  │  (BYTEA)       │
└────────────────┘  └──────────────────┘  └────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    AI Models            │
              │  - Skin Tone Analyzer   │
              │  - Clothing Classifier  │
              │  - Color Analyzer       │
              │  - Recommendation Eng.  │
              └─────────────────────────┘
```

## Database Schema

### Tables Created

1. **users** - User accounts
2. **skin_tone_analysis** - Stores user photos and skin analysis results
3. **wardrobes** - User clothing collections
4. **clothing_items** - Individual clothes with images stored as BYTEA
5. **outfit_recommendations** - Generated outfit combinations
6. **saved_outfits** - User-favorited outfits

### Image Storage Strategy

**Images are stored directly in PostgreSQL using BYTEA (binary) type:**

✅ **Advantages:**
- Atomic operations (image + metadata in one transaction)
- ACID compliance
- Backup simplicity
- No file system sync issues
- Perfect for moderate file sizes (<10MB)

**For clothing items and user photos:**
```sql
image BYTEA NOT NULL
image_filename VARCHAR(255)
image_mimetype VARCHAR(100)
```

**Retrieval:**
```typescript
GET /api/wardrobe/items/[id]/image
// Returns binary image data with proper Content-Type
```

## API Endpoints

### NextJS API Routes

#### 1. Skin Tone Analysis
```typescript
POST /api/skin-analysis
Body: FormData { photo: File, userId: string }
Response: {
  analysis: {
    fitzpatrickType: string,
    undertone: string,
    dominantColor: { r, g, b },
    confidence: number
  }
}
```

#### 2. Clothing Items
```typescript
POST /api/wardrobe/items
Body: FormData {
  image: File,
  wardrobeId: string,
  userId: string,
  name: string,
  category: string
}

GET /api/wardrobe/items?wardrobeId=xxx
Response: { items: ClothingItem[] }

GET /api/wardrobe/items/[id]/image
Response: Binary image data
```

#### 3. Outfit Recommendations
```typescript
POST /api/recommendations
Body: {
  userId: string,
  skinToneAnalysisId?: string,
  occasion: string,
  season?: string,
  count: number
}
Response: { recommendations: Outfit[] }
```

### Python AI Backend (FastAPI)

#### 1. Skin Tone Analysis
```python
POST /api/analyze-skin-tone
Body: FormData { file: File }
Response: {
  fitzpatrick_type: str,
  undertone: str,
  dominant_color: {r, g, b},
  confidence: float
}
```

#### 2. Clothing Analysis
```python
POST /api/analyze-clothing
Body: FormData { file: File }
Response: {
  clothing_type: str,
  dominant_color: {r, g, b},
  secondary_colors: [{r, g, b}],
  style: str,
  pattern: str
}
```

#### 3. Outfit Recommendations
```python
POST /api/recommend-outfits
Body: {
  skinTone: {...},
  wardrobe: [...],
  occasion: str,
  season: str
}
Response: { outfits: [...] }
```

## Setup Instructions

### Prerequisites
```bash
- Docker & Docker Compose
- Node.js 18+
- Python 3.10+ (conda environment "AI")
- PostgreSQL client (optional, for manual queries)
```

### Quick Start

1. **Run Full Stack Setup:**
```bash
cd /home/anupam/code/AIProject
./setup_full_stack.sh
```

This script will:
- Install Python dependencies (FastAPI, uvicorn, psycopg2)
- Install NextJS dependencies
- Install and configure Prisma
- Start PostgreSQL in Docker
- Run database migrations

2. **Start Services:**

Terminal 1 - Database:
```bash
docker-compose up postgres
```

Terminal 2 - AI Backend:
```bash
conda activate AI
python backend_api.py
# Runs on http://localhost:8000
```

Terminal 3 - NextJS Frontend:
```bash
cd fashion-recommender
npm run dev
# Runs on http://localhost:3000
```

### Manual Setup

1. **Install Dependencies:**
```bash
# Python backend
conda run -n AI pip install fastapi uvicorn pydantic python-multipart psycopg2-binary

# NextJS frontend
cd fashion-recommender
npm install
npm install @prisma/client prisma --save-dev
```

2. **Setup Database:**
```bash
# Start PostgreSQL
docker-compose up -d postgres

# Wait for it to be ready
sleep 10

# Run Prisma migrations
cd fashion-recommender
npx prisma generate
npx prisma db push
```

3. **Configure Environment:**
```bash
# Copy environment file
cp .env.example .env

# Edit .env with your settings
DATABASE_URL="postgresql://fashion_user:fashion_password_2024@localhost:5432/fashion_recommender"
AI_BACKEND_URL="http://localhost:8000"
```

## Database Connection

### Using Prisma (TypeScript/NextJS)

```typescript
import prisma from '@/lib/prisma'

// Create clothing item with image
const item = await prisma.clothingItem.create({
  data: {
    wardrobeId: "...",
    userId: "...",
    name: "Blue Shirt",
    image: buffer, // Binary data
    imageFilename: "shirt.jpg",
    imageMimetype: "image/jpeg",
    category: "Topwear"
  }
})

// Retrieve image
const item = await prisma.clothingItem.findUnique({
  where: { id: "..." },
  select: { image: true, imageMimetype: true }
})
```

### Direct PostgreSQL Access

```bash
# Connect to database
docker exec -it fashion_recommender_db psql -U fashion_user -d fashion_recommender

# View tables
\dt

# Query data
SELECT id, name, category FROM clothing_items;

# Check image sizes
SELECT 
  id, 
  name, 
  pg_size_pretty(length(image)) as image_size 
FROM clothing_items;
```

## Data Flow

### 1. Upload Clothing Item
```
User selects image → NextJS receives file →
Stores in database (BYTEA) → Calls AI backend →
AI analyzes image → Returns colors/style →
Updates database with AI results
```

### 2. Get Recommendations
```
User selects occasion → NextJS fetches wardrobe from DB →
Fetches skin tone analysis → Sends to AI backend →
AI generates recommendations → Saves to DB →
Returns recommendations with item details
```

### 3. Display Outfit
```
Frontend requests outfit → Fetches from recommendations table →
Includes related clothing items → For each item image:
GET /api/wardrobe/items/[id]/image → Returns binary data →
Browser displays image
```

## Performance Considerations

### Image Storage
- **Recommended**: Images < 5MB
- **Maximum**: 10MB per image
- **Optimization**: Resize images on upload to 1024x1024

### Database Indexing
```sql
-- Already created in init.sql
CREATE INDEX idx_clothing_user ON clothing_items(user_id);
CREATE INDEX idx_clothing_category ON clothing_items(category);
CREATE INDEX idx_recommendations_user ON outfit_recommendations(user_id);
```

### Caching Strategy
- Cache AI analysis results in database
- Use Next.js Image component for automatic optimization
- Consider Redis for frequently accessed images (future enhancement)

## Testing

### Test Database Connection
```bash
psql postgresql://fashion_user:fashion_password_2024@localhost:5432/fashion_recommender
```

### Test AI Backend
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

### Test NextJS API
```bash
# After starting dev server
curl http://localhost:3000/api/health
```

## Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# View logs
docker logs fashion_recommender_db

# Restart
docker-compose restart postgres
```

### Prisma Issues
```bash
# Regenerate client
npx prisma generate

# Reset database
npx prisma db push --force-reset

# View database in Prisma Studio
npx prisma studio
```

### AI Backend Issues
```bash
# Check if conda environment is activated
conda activate AI

# Test imports
python -c "import fastapi, cv2, numpy"

# Check port availability
lsof -i :8000
```

## Production Deployment

### Security
1. Change database passwords
2. Use environment variables for secrets
3. Enable HTTPS
4. Add authentication/authorization
5. Implement rate limiting

### Scaling
1. Use connection pooling (Prisma default)
2. Add Redis for caching
3. Move images to object storage (S3) if dataset grows
4. Use CDN for image delivery
5. Load balance AI backend

## File Structure

```
AIProject/
├── backend_api.py              # FastAPI server
├── docker-compose.yml          # Docker services
├── database/
│   └── init.sql               # Database schema
├── fashion-recommender/
│   ├── prisma/
│   │   └── schema.prisma      # Prisma ORM schema
│   ├── lib/
│   │   └── prisma.ts          # Prisma client
│   ├── app/
│   │   └── api/
│   │       ├── skin-analysis/ # Skin tone API
│   │       ├── wardrobe/      # Wardrobe management
│   │       └── recommendations/ # Outfit suggestions
│   └── .env                   # Environment config
├── src/                        # AI modules
│   ├── skin_tone_analyzer.py
│   ├── clothing_detector.py
│   └── recommendation_engine.py
└── models/
    └── saved_models/          # Trained models
```

## Next Steps

1. ✅ Database setup complete
2. ✅ API routes created
3. ✅ Prisma ORM configured
4. ⏳ Install dependencies
5. ⏳ Start services
6. ⏳ Test full workflow
7. ⏳ Build UI components
8. ⏳ Deploy

---

**Created:** October 2025  
**Status:** Ready for integration testing
