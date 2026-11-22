@echo off
title Employee Salary Prediction App Launcher
echo ==========================================================
echo       EMPLOYEE SALARY PREDICTION - ONE CLICK LAUNCHER
echo ==========================================================
cd /d "C:\Users\LENOVO\.gemini\antigravity\scratch\salary_prediction"

netstat -ano | findstr :8501 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo [INFO] Server is already running on port 8501!
    echo [INFO] Opening http://localhost:8501 in your browser...
    start http://localhost:8501
    exit /b 0
)

echo [INFO] Starting Streamlit server and opening browser...
start http://localhost:8501

if exist "C:\Users\LENOVO\AppData\Local\Programs\Python\Python311\python.exe" (
    "C:\Users\LENOVO\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py --server.port 8501
) else (
    python -m streamlit run app.py --server.port 8501
)

pause
