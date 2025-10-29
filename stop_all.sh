#!/bin/bash
# ============================================================
# AI Fashion Recommender - Stop All Services (Linux)
# ============================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "===================================================================="
echo "  AI FASHION RECOMMENDER - SHUTDOWN"
echo "===================================================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Stop Frontend
echo -e "${BLUE}[1/3] Stopping Frontend...${NC}"
if [ -f "$SCRIPT_DIR/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$SCRIPT_DIR/frontend.pid")
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        kill $FRONTEND_PID 2>/dev/null
        echo -e "  ${GREEN}✓ Frontend stopped (PID: $FRONTEND_PID)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Frontend not running${NC}"
    fi
    rm -f "$SCRIPT_DIR/frontend.pid"
else
    # Try to kill by port
    FRONTEND_PID=$(lsof -t -i:3000 2>/dev/null)
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo -e "  ${GREEN}✓ Frontend stopped (PID: $FRONTEND_PID)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Frontend not running${NC}"
    fi
fi

# Stop Backend
echo ""
echo -e "${BLUE}[2/3] Stopping Backend...${NC}"
if [ -f "$SCRIPT_DIR/backend.pid" ]; then
    BACKEND_PID=$(cat "$SCRIPT_DIR/backend.pid")
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        kill $BACKEND_PID 2>/dev/null
        echo -e "  ${GREEN}✓ Backend stopped (PID: $BACKEND_PID)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Backend not running${NC}"
    fi
    rm -f "$SCRIPT_DIR/backend.pid"
else
    # Try to kill by port
    BACKEND_PID=$(lsof -t -i:8000 2>/dev/null)
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo -e "  ${GREEN}✓ Backend stopped (PID: $BACKEND_PID)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Backend not running${NC}"
    fi
fi

# Stop Database
echo ""
echo -e "${BLUE}[3/3] Stopping Database...${NC}"
cd "$SCRIPT_DIR"

if docker ps --format "{{.Names}}" | grep -q "fashion_recommender_db"; then
    docker stop fashion_recommender_db >/dev/null 2>&1
    echo -e "  ${GREEN}✓ Database stopped${NC}"
else
    echo -e "  ${YELLOW}⚠ Database not running${NC}"
fi

# Clean up log files (optional)
if [ -f "$SCRIPT_DIR/backend.log" ] || [ -f "$SCRIPT_DIR/frontend.log" ]; then
    echo ""
    read -p "Delete log files? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$SCRIPT_DIR/backend.log" "$SCRIPT_DIR/frontend.log"
        echo -e "  ${GREEN}✓ Log files deleted${NC}"
    fi
fi

echo ""
echo "===================================================================="
echo -e "  ${GREEN}SHUTDOWN COMPLETE!${NC}"
echo "===================================================================="
echo ""
echo "  All services stopped."
echo "  Database container is stopped but not removed."
echo ""
echo "  To start again, run:"
echo -e "  ${YELLOW}./start_all.sh${NC}"
echo ""
echo "  To completely remove database (delete data), run:"
echo -e "  ${YELLOW}docker-compose down -v${NC}"
echo ""
echo "===================================================================="
echo ""
