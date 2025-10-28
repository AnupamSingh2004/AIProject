@echo off
REM ============================================================
REM AI Fashion Recommender - Shutdown Script
REM ============================================================

echo.
echo ====================================================================
echo   AI FASHION RECOMMENDER - SHUTDOWN
echo ====================================================================
echo.

echo [1/3] Stopping Docker containers...
docker stop fashion_recommender_db >nul 2>&1
echo   OK: Database stopped

echo.
echo [2/3] Closing backend and frontend windows...
taskkill /FI "WindowTitle eq AI Fashion Backend*" /F >nul 2>&1
taskkill /FI "WindowTitle eq AI Fashion Frontend*" /F >nul 2>&1
echo   OK: Services stopped

echo.
echo [3/3] Cleanup complete!
echo.
echo ====================================================================
echo   ALL SERVICES STOPPED
echo ====================================================================
echo.
pause
