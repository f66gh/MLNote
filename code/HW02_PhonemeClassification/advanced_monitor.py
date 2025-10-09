#!/usr/bin/env python3
"""
高级系统监控脚本
监控CPU、GPU、内存使用情况
"""

import time
import psutil
import torch
import subprocess
import json
from datetime import datetime

def get_gpu_info():
    """获取详细的GPU信息"""
    try:
        # 使用nvidia-smi获取GPU信息
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            gpu_data = result.stdout.strip().split(', ')
            return {
                'name': gpu_data[0],
                'gpu_util': int(gpu_data[1]),
                'mem_util': int(gpu_data[2]),
                'mem_used': int(gpu_data[3]),
                'mem_total': int(gpu_data[4]),
                'temperature': int(gpu_data[5]),
                'power': float(gpu_data[6]),
                'clock_graphics': int(gpu_data[7]),
                'clock_memory': int(gpu_data[8])
            }
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
    return None

def get_cpu_info():
    """获取CPU信息"""
    return {
        'usage': psutil.cpu_percent(interval=1),
        'count': psutil.cpu_count(),
        'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        'temp': get_cpu_temp()
    }

def get_cpu_temp():
    """获取CPU温度（Windows）"""
    try:
        # Windows上获取CPU温度比较复杂，这里返回占位符
        return "N/A"
    except:
        return "N/A"

def get_memory_info():
    """获取内存信息"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / 1024**3,
        'used': mem.used / 1024**3,
        'percent': mem.percent,
        'available': mem.available / 1024**3
    }

def monitor_loop():
    """监控循环"""
    print("🚀 高级系统监控启动")
    print("按 Ctrl+C 停止监控")
    print("💡 监控信息将累积显示，新信息在下方")
    print("=" * 80)
    
    try:
        while True:
            # 不清屏，让信息累积显示
            # subprocess.run('cls', shell=True)  # 注释掉清屏命令
            
            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n📅 [{current_time}] 系统状态:")
            print("-" * 60)
            
            # CPU信息
            cpu_info = get_cpu_info()
            mem_info = get_memory_info()
            gpu_info = get_gpu_info()
            
            # 简化的一行显示格式
            cpu_status = f"💻 CPU: {cpu_info['usage']:.1f}%"
            mem_status = f"💾 内存: {mem_info['used']:.1f}GB/{mem_info['total']:.1f}GB ({mem_info['percent']:.1f}%)"
            
            if gpu_info:
                gpu_status = f"🎮 GPU: {gpu_info['gpu_util']}% | 显存: {gpu_info['mem_used']}MB/{gpu_info['mem_total']}MB | 温度: {gpu_info['temperature']}°C | 功耗: {gpu_info['power']:.1f}W"
                print(f"{cpu_status} | {mem_status}")
                print(f"{gpu_status}")
            else:
                print(f"{cpu_status} | {mem_status} | 🎮 GPU: 获取失败")
            
            # PyTorch GPU信息（如果可用）
            if torch.cuda.is_available():
                torch_gpu_mem = torch.cuda.memory_allocated() / 1024**3
                torch_gpu_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"🔥 PyTorch: 已分配 {torch_gpu_mem:.2f}GB | 已缓存 {torch_gpu_cached:.2f}GB")
            
            # 添加状态指示器
            status_indicators = []
            if gpu_info:
                if gpu_info['gpu_util'] > 90:
                    status_indicators.append("🟢 GPU高负载")
                elif gpu_info['gpu_util'] < 50:
                    status_indicators.append("🟡 GPU低负载")
                
                if gpu_info['temperature'] > 80:
                    status_indicators.append("🔥 GPU高温")
                
                if mem_info['percent'] > 80:
                    status_indicators.append("⚠️ 内存紧张")
            
            if status_indicators:
                print(f"📊 状态: {' | '.join(status_indicators)}")
            else:
                print("📊 状态: 🟢 运行正常")
            
            # 等待2秒
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    monitor_loop()

