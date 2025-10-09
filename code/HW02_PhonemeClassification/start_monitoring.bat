@echo off
echo 启动多窗口监控...
echo.
echo 窗口1: 训练程序 (手动启动)
echo 窗口2: GPU监控 (即将启动)
echo 窗口3: 系统监控 (即将启动)
echo.
pause

REM 启动GPU监控
start "GPU监控" cmd /k "cd /d %~dp0 && monitor_gpu.bat"

REM 等待2秒
timeout /t 2 /nobreak >nul

REM 启动高级监控
start "系统监控" cmd /k "cd /d %~dp0 && C:\Users\BaiMu\anaconda3\envs\myenv\python.exe advanced_monitor.py"

echo 监控窗口已启动！
echo 现在可以在主窗口运行训练程序了。
pause
