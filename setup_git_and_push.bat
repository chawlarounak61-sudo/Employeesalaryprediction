@echo off
title Connect Project to Remote GitHub Repository
echo ==========================================================
echo        CONNECT PROJECT TO REMOTE GITHUB REPOSITORY
echo ==========================================================
echo.
cd /d "%~dp0"

set "PATH=%PATH%;C:\Program Files\Git\cmd;C:\Program Files\Git\bin"

where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git could not be located.
    pause
    exit /b 1
)

echo [OK] Git is ready:
git --version
echo.

set /p REPO_URL="Enter your remote GitHub repository URL (e.g. https://github.com/username/repo.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] No repository URL entered. Exiting.
    pause
    exit /b 1
)

echo.
echo [1/4] Ensuring files are staged and committed...
git add .
git commit -m "Update: Enterprise Employee Salary Prediction System" >nul 2>&1

echo [2/4] Setting main branch...
git branch -M main

echo [3/4] Setting remote origin...
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%

echo [4/4] Pushing code to GitHub (main branch)...
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
    echo [INFO] If GitHub prompted for login in your browser, please
    echo approve the sign-in and run: git push -u origin main
)

echo.
pause
