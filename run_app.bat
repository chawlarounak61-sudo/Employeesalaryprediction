@echo off
cd /d "%~dp0"
echo Starting Employee Salary Prediction Web App...
start http://localhost:8501
python -m streamlit run app.py
pause
