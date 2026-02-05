@echo off
echo ========================================
echo Network Intrusion Detection System
echo Setup Verification
echo ========================================
echo.

echo [1/5] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo ✓ Python found
echo.

echo [2/5] Checking Node.js installation...
node --version
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found! Please install Node.js 16+
    pause
    exit /b 1
)
echo ✓ Node.js found
echo.

echo [3/5] Checking dataset...
if exist "KDDTrain+.txt" (
    echo ✓ Dataset found: KDDTrain+.txt
) else (
    echo WARNING: Dataset not found!
    echo Please ensure KDDTrain+.txt is in the project directory
)
echo.

echo [4/5] Checking project structure...
if exist "src" (
    echo ✓ src/ directory found
) else (
    echo ERROR: src/ directory not found!
)

if exist "frontend" (
    echo ✓ frontend/ directory found
) else (
    echo ERROR: frontend/ directory not found!
)

if exist "config.yaml" (
    echo ✓ config.yaml found
) else (
    echo ERROR: config.yaml not found!
)
echo.

echo [5/5] Checking dependencies...
echo Checking Python packages...
python -c "import pandas, numpy, sklearn, fastapi" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Some Python packages missing
    echo Run: pip install -r requirements.txt
) else (
    echo ✓ Core Python packages installed
)
echo.

echo ========================================
echo Setup Verification Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Install dependencies:
echo    pip install -r requirements.txt
echo    cd frontend ^&^& npm install
echo.
echo 2. Train models:
echo    python train.py
echo.
echo 3. Start the application:
echo    Terminal 1: python -m uvicorn src.api:app --reload
echo    Terminal 2: cd frontend ^&^& npm start
echo.
echo See QUICKSTART.md for detailed instructions
echo.
pause
