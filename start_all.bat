@echo off
REM ============================================================
REM AI Fashion Recommender - Complete Startup Script
REM ============================================================

echo.
echo ====================================================================
echo   AI FASHION RECOMMENDER - STARTUP
echo ====================================================================
echo.

REM Check if Docker is running
echo [1/5] Checking Docker...
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: Docker is not running!
    echo   Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo   OK: Docker is running

REM Check if database container exists
echo.
echo [2/5] Starting PostgreSQL Database...
docker ps -a --format "{{.Names}}" | findstr "fashion_recommender_db" >nul
if %ERRORLEVEL% NEQ 0 (
    echo   Creating new database container...
    cd /d "%~dp0"
    docker-compose up -d
) else (
    docker start fashion_recommender_db >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo   OK: Database started
    ) else (
        echo   Starting database with docker-compose...
        docker-compose up -d
    )
)

REM Wait for database to be ready
echo   Waiting for database to be ready...
timeout /t 3 /nobreak >nul
echo   OK: Database is ready

REM Start Backend API (Python FastAPI)
echo.
echo [3/5] Starting Backend API Server...
cd /d "%~dp0backend"
start "AI Fashion Backend" cmd /k "python start_backend.py"
echo   OK: Backend starting on http://localhost:8000
timeout /t 5 /nobreak >nul

REM Start Frontend (Next.js)
echo.
echo [4/5] Starting Frontend Application...
cd /d "%~dp0fashion-recommender"
start "AI Fashion Frontend" cmd /k "npm run dev"
echo   OK: Frontend starting on http://localhost:3000
timeout /t 5 /nobreak >nul

REM Open browser
echo.
echo [5/5] Opening application in browser...
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo ====================================================================
echo   STARTUP COMPLETE!
echo ====================================================================
echo.
echo   Services Running:
echo   - Database:  PostgreSQL on localhost:5432
echo   - Backend:   FastAPI on http://localhost:8000
echo   - Frontend:  Next.js on http://localhost:3000
echo   - API Docs:  http://localhost:8000/docs
echo.
echo   Press any key to see this window's status...
echo   Close this window to keep services running.
echo ====================================================================
echo.
pause
