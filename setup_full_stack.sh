#!/bin/bash
# Complete Setup Script for Fashion Recommender with Database Integration

set -e

echo "======================================================================"
echo "🚀 Fashion Recommender - Complete Setup with Database"
echo "======================================================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${BLUE}Step 1: Installing Python Backend Dependencies${NC}"
cd /home/anupam/code/AIProject
conda run -n AI pip install fastapi uvicorn pydantic python-multipart psycopg2-binary sqlalchemy

echo -e "\n${BLUE}Step 2: Installing NextJS Dependencies${NC}"
cd fashion-recommender
npm install
npm install @prisma/client
npm install prisma --save-dev

echo -e "\n${BLUE}Step 3: Setting up Database with Docker${NC}"
cd ..
docker-compose up -d postgres

echo -e "\n${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
sleep 10

echo -e "\n${BLUE}Step 4: Running Prisma Migrations${NC}"
cd fashion-recommender
npx prisma generate
npx prisma db push

echo -e "\n${GREEN}✅ Setup Complete!${NC}"
echo ""
echo "======================================================================"
echo "📋 Next Steps:"
echo "======================================================================"
echo ""
echo "1. Start the AI Backend:"
echo "   conda run -n AI python backend_api.py"
echo ""
echo "2. Start the NextJS App (in another terminal):"
echo "   cd fashion-recommender && npm run dev"
echo ""
echo "3. Access the application:"
echo "   Frontend: http://localhost:3000"
echo "   AI Backend: http://localhost:8000"
echo "   Database: postgresql://localhost:5432/fashion_recommender"
echo ""
echo "======================================================================"
