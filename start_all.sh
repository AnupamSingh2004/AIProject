#!/bin/bash
# ============================================================
# AI Fashion Recommender - Complete Startup Script (Linux)
# ============================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "===================================================================="
echo "  AI FASHION RECOMMENDER - STARTUP"
echo "===================================================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Docker is running
echo -e "${BLUE}[1/5] Checking Docker...${NC}"
if ! docker info >/dev/null 2>&1; then
    echo -e "  ${RED}ERROR: Docker is not running!${NC}"
    echo -e "  Please start Docker and try again."
    echo -e "  Run: ${YELLOW}sudo systemctl start docker${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Docker is running${NC}"

# Start PostgreSQL Database
echo ""
echo -e "${BLUE}[2/5] Starting PostgreSQL Database...${NC}"
cd "$SCRIPT_DIR"

# Check if container exists
if docker ps -a --format "{{.Names}}" | grep -q "fashion_recommender_db"; then
    # Container exists, try to start it
    if docker start fashion_recommender_db >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Database started${NC}"
    else
        echo "  Starting database with docker-compose..."
        docker-compose up -d postgres
    fi
else
    # Container doesn't exist, create it
    echo "  Creating new database container..."
    docker-compose up -d postgres
fi

# Wait for database to be ready
echo "  Waiting for database to be ready..."
sleep 5
echo -e "  ${GREEN}✓ Database is ready${NC}"

# Start Backend API (Python FastAPI)
echo ""
echo -e "${BLUE}[3/5] Starting Backend API Server...${NC}"
cd "$SCRIPT_DIR/backend"

# Check if backend is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "  ${YELLOW}⚠ Backend already running on port 8000${NC}"
else
    # Start backend in background
    if [ -f "start_backend.py" ]; then
        nohup python start_backend.py > "$SCRIPT_DIR/backend.log" 2>&1 &
        BACKEND_PID=$!
        echo $BACKEND_PID > "$SCRIPT_DIR/backend.pid"
        echo -e "  ${GREEN}✓ Backend starting on http://localhost:8000 (PID: $BACKEND_PID)${NC}"
    else
        nohup python main.py > "$SCRIPT_DIR/backend.log" 2>&1 &
        BACKEND_PID=$!
        echo $BACKEND_PID > "$SCRIPT_DIR/backend.pid"
        echo -e "  ${GREEN}✓ Backend starting on http://localhost:8000 (PID: $BACKEND_PID)${NC}"
    fi
fi
sleep 3

# Start Frontend (Next.js)
echo ""
echo -e "${BLUE}[4/5] Starting Frontend Application...${NC}"
cd "$SCRIPT_DIR/fashion-recommender"

# Check if frontend is already running
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "  ${YELLOW}⚠ Frontend already running on port 3000${NC}"
else
    # Start frontend in background
    nohup npm run dev > "$SCRIPT_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$SCRIPT_DIR/frontend.pid"
    echo -e "  ${GREEN}✓ Frontend starting on http://localhost:3000 (PID: $FRONTEND_PID)${NC}"
fi
sleep 5

# Open browser
echo ""
echo -e "${BLUE}[5/5] Opening application in browser...${NC}"
sleep 2

# Try to open browser (works on most Linux systems)
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://localhost:3000 >/dev/null 2>&1 &
elif command -v gnome-open >/dev/null 2>&1; then
    gnome-open http://localhost:3000 >/dev/null 2>&1 &
elif command -v kde-open >/dev/null 2>&1; then
    kde-open http://localhost:3000 >/dev/null 2>&1 &
fi

echo ""
echo "===================================================================="
echo -e "  ${GREEN}STARTUP COMPLETE!${NC}"
echo "===================================================================="
echo ""
echo "  Services Running:"
echo "  - Database:  PostgreSQL on localhost:5432"
echo "  - Backend:   FastAPI on http://localhost:8000"
echo "  - Frontend:  Next.js on http://localhost:3000"
echo "  - API Docs:  http://localhost:8000/docs"
echo ""
echo "  Logs:"
echo "  - Backend:   $SCRIPT_DIR/backend.log"
echo "  - Frontend:  $SCRIPT_DIR/frontend.log"
echo ""
echo "  To stop all services, run:"
echo -e "  ${YELLOW}./stop_all.sh${NC}"
echo ""
echo "===================================================================="
echo ""
