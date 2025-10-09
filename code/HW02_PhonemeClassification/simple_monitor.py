#!/usr/bin/env python3
"""
简洁版系统监控 - 累积显示
"""

import time
import psutil
import torch
import subprocess
from datetime import datetime

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            data = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(data[0]),
                'mem_util': int(data[1]), 
                'mem_used': int(data[2]),
                'mem_total': int(data[3]),
                'temperature': int(data[4]),
                'power': float(data[5])
            }
    except:
        pass
    return None

def monitor_simple():
    """简洁监控循环"""
    print("🚀 简洁监控启动 - 累积显示模式")
    print("时间格式: [HH:MM:SS] CPU% | 内存GB | GPU% | 显存MB | 温度°C | 功耗W | 状态")
    print("-" * 80)
    
    try:
        while True:
            # 获取时间
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # 获取系统信息
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            mem_used = memory.used / 1024**3
            
            # 获取GPU信息
            gpu_info = get_gpu_info()
            
            if gpu_info:
                # PyTorch GPU内存
                torch_mem = 0
                if torch.cuda.is_available():
                    torch_mem = torch.cuda.memory_allocated() / 1024**2  # MB
                
                # 状态判断
                status = "🟢"
                if gpu_info['gpu_util'] < 50:
                    status = "🟡"
                elif gpu_info['temperature'] > 80:
                    status = "🔥"
                elif memory.percent > 85:
                    status = "⚠️"
                
                print(f"[{current_time}] CPU:{cpu_percent:5.1f}% | 内存:{mem_used:5.1f}GB | GPU:{gpu_info['gpu_util']:3d}% | 显存:{gpu_info['mem_used']:5d}MB | 温度:{gpu_info['temperature']:2d}°C | 功耗:{gpu_info['power']:5.1f}W | PyTorch:{torch_mem:5.0f}MB {status}")
            else:
                print(f"[{current_time}] CPU:{cpu_percent:5.1f}% | 内存:{mem_used:5.1f}GB | GPU:N/A")
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    monitor_simple()
