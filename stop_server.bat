@echo off
echo Stopping Salary Prediction Streamlit Server...
schtasks /end /tn "SalaryPredictionServer" >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
)
echo Server stopped.
