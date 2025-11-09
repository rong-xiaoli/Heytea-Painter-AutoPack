@echo off
setlocal enabledelayedexpansion
title Heytea Painter
cd /d "%~dp0"

REM Check if env folder exists
if not exist "env\" (
    echo [ERROR] 环境文件夹不存在！
    echo 请先运行 setup_env.bat 创建环境
    echo.
    pause
    exit /b 1
)

REM Check if Python is in env
if not exist "env\python.exe" (
    echo [ERROR] Python 未找到！
    echo 环境可能损坏，请重新运行 setup_env.bat
    echo.
    pause
    exit /b 1
)

echo 启动 Heytea Painter...
echo.

REM Use env Python directly
start "" "%~dp0env\python.exe" heytea_modern.py

REM Exit immediately (window will close)
exit
