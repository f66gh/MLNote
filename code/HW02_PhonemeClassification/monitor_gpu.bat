@echo off
echo ========================================
echo        GPU 性能实时监控
echo ========================================
echo 按 Ctrl+C 停止监控
echo.

:loop
cls
echo [%date% %time%] GPU 状态监控
echo ========================================
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits
echo.
echo 详细信息:
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv
echo.
timeout /t 2 /nobreak >nul
goto loop

