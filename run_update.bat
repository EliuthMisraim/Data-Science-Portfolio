@echo off
PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update_portfolio.ps1"
if %ERRORLEVEL% neq 0 (
    echo.
    pause
)
