@echo off
title Connect Project to Remote GitHub Repository
echo ==========================================================
echo        CONNECT PROJECT TO REMOTE GITHUB REPOSITORY
echo ==========================================================
echo.
cd /d "%~dp0"

where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in your PATH.
    echo.
    echo Please install Git for Windows from:
    echo https://git-scm.com/download/win
    echo.
    echo Opening Git download page in your browser...
    start https://git-scm.com/download/win
    echo.
    echo After installing Git, restart this script!
    pause
    exit /b 1
)

echo [OK] Git is installed.
echo.

set /p REPO_URL="Enter your remote GitHub repository URL (e.g. https://github.com/username/repo.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] No repository URL entered. Exiting.
    pause
    exit /b 1
)

echo.
echo [1/5] Initializing Git repository...
if not exist ".git" (
    git init
)

echo [2/5] Staging files...
git add .

echo [3/5] Creating commit...
git commit -m "Initial commit: Enterprise Employee Salary Prediction System (SalaryIQ)"

echo [4/5] Setting main branch and remote origin...
git branch -M main
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%

echo [5/5] Pushing to GitHub...
echo.
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ==========================================================
    echo [SUCCESS] Your project has been connected and pushed to GitHub!
    echo Remote: %REPO_URL%
    echo ==========================================================
) else (
    echo.
    echo [NOTE] If this is your first time connecting, GitHub may ask
    echo you to authenticate in the browser or terminal.
    echo You can also re-run: git push -u origin main
)

echo.
pause
