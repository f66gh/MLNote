# 模型对比训练脚本
# 运行此脚本来比较原始模型和改进模型的性能

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# 导入所有必要的函数和类
with open('index.py', 'r', encoding='utf-8') as f:
    exec(f.read())

def compare_models():
    """比较原始模型和改进模型的性能"""
    
    # 数据路径
    tr_path = 'covid.train.csv'
    tt_path = 'covid.test.csv'
    
    print("="*60)
    print("模型性能对比实验")
    print("="*60)
    
    # 1. 原始模型配置
    print("\n训练原始模型...")
    simple_config = {
        'n_epochs': 1000,
        'batch_size': 270,
        'optimizer': 'SGD',
        'optim_hparas': {'lr': 0.001, 'momentum': 0.9},
        'early_stop': 200,
        'save_path': 'models/simple_model.pth'
    }
    
    # 简单网络结构
    class SimpleNet(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.criterion = nn.MSELoss(reduction='mean')
        
        def forward(self, x):
            return self.net(x).squeeze(1)
        
        def cal_loss(self, pred, target):
            return self.criterion(pred, target)
    
    # 训练简单模型
    tr_set = prep_dataloader(tr_path, 'train', simple_config['batch_size'], target_only=False)
    dv_set = prep_dataloader(tr_path, 'dev', simple_config['batch_size'], target_only=False)
    
    simple_model = SimpleNet(tr_set.dataset.dim).to(get_device())
    simple_loss, simple_record = train(tr_set, dv_set, simple_model, simple_config, get_device())
    
    print(f"原始模型最终验证损失: {simple_loss:.4f}")
    
    # 2. 改进模型配置
    print("\n训练改进模型...")
    improved_config = {
        'n_epochs': 1000,
        'batch_size': 64,
        'optimizer': 'Adam',
        'optim_hparas': {'lr': 0.001, 'weight_decay': 1e-5},
        'early_stop': 200,
        'save_path': 'models/improved_model.pth'
    }
    
    # 训练改进模型
    tr_set_improved = prep_dataloader(tr_path, 'train', improved_config['batch_size'], target_only=False)
    dv_set_improved = prep_dataloader(tr_path, 'dev', improved_config['batch_size'], target_only=False)
    
    improved_model = NeuralNet(tr_set_improved.dataset.dim).to(get_device())
    improved_loss, improved_record = train(tr_set_improved, dv_set_improved, improved_model, improved_config, get_device())
    
    print(f"改进模型最终验证损失: {improved_loss:.4f}")
    
    # 3. 绘制对比图
    print("\n生成对比图表...")
    
    # 学习曲线对比
    figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(simple_record['train'], label='简单模型-训练', color='red', alpha=0.7)
    plt.plot(simple_record['dev'], label='简单模型-验证', color='red', linestyle='--')
    plt.plot(improved_record['train'], label='改进模型-训练', color='blue', alpha=0.7)
    plt.plot(improved_record['dev'], label='改进模型-验证', color='blue', linestyle='--')
    plt.xlabel('训练步数')
    plt.ylabel('MSE损失')
    plt.title('学习曲线对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 性能对比柱状图
    plt.subplot(1, 2, 2)
    models = ['原始模型', '改进模型']
    losses = [simple_loss, improved_loss]
    colors = ['red', 'blue']
    
    bars = plt.bar(models, losses, color=colors, alpha=0.7)
    plt.ylabel('验证损失')
    plt.title('模型性能对比')
    
    # 添加数值标签
    for bar, loss in zip(bars, losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 输出改进总结
    improvement = ((simple_loss - improved_loss) / simple_loss) * 100
    print("\n="*60)
    print("实验结果总结:")
    print("="*60)
    print(f"原始模型验证损失: {simple_loss:.4f}")
    print(f"改进模型验证损失: {improved_loss:.4f}")
    print(f"性能改进: {improvement:.2f}%")
    
    if improvement > 0:
        print("✅ 模型性能有所提升！")
    else:
        print("❌ 模型性能未能提升，建议进一步调优")
    
    print("\n主要改进点:")
    print("1. 网络结构: 单层 → 多层深度网络")
    print("2. 正则化: 无 → Dropout + BatchNorm + L2正则化")
    print("3. 优化器: SGD → Adam")
    print("4. 批次大小: 270 → 64")

if __name__ == "__main__":
    compare_models()
