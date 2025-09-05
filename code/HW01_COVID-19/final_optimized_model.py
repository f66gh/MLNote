# 最终优化版本 - COVID-19预测模型
# 基于train_comparison.py进行全面优化
# 修复原代码错误并实现简单/中阶/强基线

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F

# For data preprocess
import numpy as np
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def get_device():
    '''自动选择计算设备'''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    '''绘制学习曲线'''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

class COVID19Dataset(Dataset):
    '''优化的COVID19数据集类 - 修复原始错误'''
    def __init__(self, path, mode='train', target_only=False, feature_type='all', scaler=None):
        self.mode = mode
        self.scaler = scaler
        
        # 读取数据
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        # 特征选择 - 修复原代码中的pass错误
        if feature_type == 'simple':
            # 简单基线：使用所有特征
            feats = list(range(93))
        elif feature_type == 'medium':
            # 中阶基线：40个州 + 2个tested_positive特征 (indices = 57 & 75)
            feats = list(range(40)) + [57, 75]  # 修复原代码错误
        elif feature_type == 'advanced':
            # 强基线：特征工程 - 选择更多有用特征
            # 40个州 + 症状相关特征 + 测试相关特征 + 时间趋势特征
            feats = (list(range(40)) +           # 40个州特征
                    [57, 75] +                   # tested_positive特征
                    [41, 42, 43, 44, 45, 46] +   # 症状特征(咳嗽、发热等)
                    [47, 48, 49] +               # 接触史特征
                    [50, 51, 52])                # 其他相关特征
        else:
            # 默认使用所有特征
            feats = list(range(93))
            
        if mode == 'test':
            # 测试数据
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # 训练数据
            target = data[:, -1]
            data = data[:, feats]
            
            # 修复数据分割错误 - 使用sklearn的train_test_split代替简单的模运算
            if mode == 'train':
                train_data, _, train_target, _ = train_test_split(
                    data, target, test_size=0.1, random_state=myseed, stratify=None)
                self.data = torch.FloatTensor(train_data)
                self.target = torch.FloatTensor(train_target)
            elif mode == 'dev':
                _, dev_data, _, dev_target = train_test_split(
                    data, target, test_size=0.1, random_state=myseed, stratify=None)
                self.data = torch.FloatTensor(dev_data)
                self.target = torch.FloatTensor(dev_target)
        
        # 修复标准化错误 - 只在训练集上计算统计量，然后应用到所有集合
        if mode == 'train' and self.scaler is None:
            # 在训练集上拟合scaler
            self.scaler = StandardScaler()
            # 只标准化非州特征（州特征已经是one-hot编码）
            if self.data.shape[1] > 40:
                self.data[:, 40:] = torch.FloatTensor(
                    self.scaler.fit_transform(self.data[:, 40:].numpy()))
        elif self.scaler is not None and self.data.shape[1] > 40:
            # 在验证集和测试集上使用训练集的统计量
            self.data[:, 40:] = torch.FloatTensor(
                self.scaler.transform(self.data[:, 40:].numpy()))
        
        self.dim = self.data.shape[1]
        print(f'Finished reading the {mode} set of COVID19 Dataset ({len(self.data)} samples found, each dim = {self.dim})')

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

def prep_dataloader(path, mode, batch_size, feature_type='all', scaler=None):
    '''准备数据加载器'''
    dataset = COVID19Dataset(path, mode=mode, feature_type=feature_type, scaler=scaler)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=0, pin_memory=True)
    return dataloader, dataset.scaler if mode == 'train' else scaler

# 简单基线模型
class SimpleNet(nn.Module):
    '''简单基线：基础的两层网络'''
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

# 中阶基线模型
class MediumNet(nn.Module):
    '''中阶基线：添加dropout和更深的网络'''
    def __init__(self, input_dim):
        super(MediumNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.criterion = nn.MSELoss(reduction='mean')
    
    def forward(self, x):
        return self.net(x).squeeze(1)
    
    def cal_loss(self, pred, target):
        return self.criterion(pred, target)

# 强基线模型
class AdvancedNet(nn.Module):
    '''强基线：深度网络 + BatchNorm + Dropout + 残差连接 + L2正则化'''
    def __init__(self, input_dim):
        super(AdvancedNet, self).__init__()
        
        # 主干网络
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 残差连接的投影层
        self.shortcut = nn.Linear(256, 256)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        self.criterion = nn.MSELoss(reduction='mean')
    
    def forward(self, x):
        # 主干前向传播
        x1 = self.layer1(x)           # input_dim -> 256
        x2 = self.layer2(x1)          # 256 -> 512
        x3 = self.layer3(x2)          # 512 -> 256
        
        # 残差连接 (x1 + x3)
        shortcut = self.shortcut(x1)  # 保证维度匹配
        x3_residual = x3 + shortcut   # 残差连接
        
        x4 = self.layer4(x3_residual) # 256 -> 128
        output = self.output(x4)       # 128 -> 1
        
        return output.squeeze(1)
    
    def cal_loss(self, pred, target, l2_lambda=1e-4):
        '''计算损失 + L2正则化'''
        mse_loss = self.criterion(pred, target)
        
        # L2正则化
        l2_penalty = 0
        for param in self.parameters():
            l2_penalty += torch.norm(param, p=2)
        
        total_loss = mse_loss + l2_lambda * l2_penalty
        return total_loss

def train(tr_set, dv_set, model, config, device):
    '''训练函数 - 添加学习率调度'''
    n_epochs = config['n_epochs']
    
    # 设置优化器
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    
    # 学习率调度器
    if config.get('scheduler'):
        if config['scheduler']['type'] == 'StepLR':
            scheduler = StepLR(optimizer, step_size=config['scheduler']['step_size'], 
                             gamma=config['scheduler']['gamma'])
        elif config['scheduler']['type'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    else:
        scheduler = None
    
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # 根据模型类型计算损失
            if isinstance(model, AdvancedNet):
                mse_loss = model.cal_loss(pred, y, l2_lambda=config.get('l2_lambda', 1e-4))
            else:
                mse_loss = model.cal_loss(pred, y)
                
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())
        
        # 验证
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        
        epoch += 1
        loss_record['dev'].append(dev_mse)
        
        # 学习率调度
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(dev_mse)
            else:
                scheduler.step()
        
        if early_stop_cnt > config['early_stop']:
            break
    
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

def dev(dv_set, model, device):
    '''验证函数'''
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            if isinstance(model, AdvancedNet):
                mse_loss = model.cal_loss(pred, y, l2_lambda=0)  # 验证时不加正则化
            else:
                mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)
    return total_loss

def comprehensive_comparison():
    '''全面的模型对比实验'''
    
    tr_path = 'covid.train.csv'
    tt_path = 'covid.test.csv'
    device = get_device()
    os.makedirs('models', exist_ok=True)
    
    print("="*80)
    print("COVID-19预测模型全面对比实验")
    print("简单基线 vs 中阶基线 vs 强基线")
    print("="*80)
    
    results = {}
    
    # 1. 简单基线
    print("\n1. 训练简单基线模型...")
    simple_config = {
        'n_epochs': 800,
        'batch_size': 270,
        'optimizer': 'SGD',
        'optim_hparas': {'lr': 0.001, 'momentum': 0.9},
        'early_stop': 150,
        'save_path': 'models/simple_baseline.pth'
    }
    
    tr_set, scaler = prep_dataloader(tr_path, 'train', simple_config['batch_size'], 'simple')
    dv_set, _ = prep_dataloader(tr_path, 'dev', simple_config['batch_size'], 'simple', scaler)
    
    simple_model = SimpleNet(tr_set.dataset.dim).to(device)
    simple_loss, simple_record = train(tr_set, dv_set, simple_model, simple_config, device)
    results['simple'] = {'loss': simple_loss, 'record': simple_record, 'features': tr_set.dataset.dim}
    
    # 2. 中阶基线  
    print("\n2. 训练中阶基线模型（特征选择 + 改进架构）...")
    medium_config = {
        'n_epochs': 1000,
        'batch_size': 128,
        'optimizer': 'Adam',
        'optim_hparas': {'lr': 0.001, 'weight_decay': 1e-5},
        'early_stop': 200,
        'save_path': 'models/medium_baseline.pth'
    }
    
    tr_set_med, scaler_med = prep_dataloader(tr_path, 'train', medium_config['batch_size'], 'medium')
    dv_set_med, _ = prep_dataloader(tr_path, 'dev', medium_config['batch_size'], 'medium', scaler_med)
    
    medium_model = MediumNet(tr_set_med.dataset.dim).to(device)
    medium_loss, medium_record = train(tr_set_med, dv_set_med, medium_model, medium_config, device)
    results['medium'] = {'loss': medium_loss, 'record': medium_record, 'features': tr_set_med.dataset.dim}
    
    # 3. 强基线
    print("\n3. 训练强基线模型（深度网络 + 全面优化）...")
    advanced_config = {
        'n_epochs': 1200,
        'batch_size': 64,
        'optimizer': 'Adam',
        'optim_hparas': {'lr': 0.001, 'weight_decay': 1e-4},
        'early_stop': 300,
        'l2_lambda': 1e-4,
        'scheduler': {
            'type': 'ReduceLROnPlateau'
        },
        'save_path': 'models/advanced_baseline.pth'
    }
    
    tr_set_adv, scaler_adv = prep_dataloader(tr_path, 'train', advanced_config['batch_size'], 'advanced')
    dv_set_adv, _ = prep_dataloader(tr_path, 'dev', advanced_config['batch_size'], 'advanced', scaler_adv)
    
    advanced_model = AdvancedNet(tr_set_adv.dataset.dim).to(device)
    advanced_loss, advanced_record = train(tr_set_adv, dv_set_adv, advanced_model, advanced_config, device)
    results['advanced'] = {'loss': advanced_loss, 'record': advanced_record, 'features': tr_set_adv.dataset.dim}
    
    # 4. 结果可视化和分析
    print("\n4. 生成对比图表...")
    figure(figsize=(15, 5))
    
    # 学习曲线对比
    plt.subplot(1, 3, 1)
    plt.plot(results['simple']['record']['dev'], label='简单基线', color='red', alpha=0.8)
    plt.plot(results['medium']['record']['dev'], label='中阶基线', color='blue', alpha=0.8)
    plt.plot(results['advanced']['record']['dev'], label='强基线', color='green', alpha=0.8)
    plt.xlabel('训练轮次')
    plt.ylabel('验证损失')
    plt.title('学习曲线对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 性能对比
    plt.subplot(1, 3, 2)
    models = ['简单基线', '中阶基线', '强基线']
    losses = [results['simple']['loss'], results['medium']['loss'], results['advanced']['loss']]
    colors = ['red', 'blue', 'green']
    
    bars = plt.bar(models, losses, color=colors, alpha=0.7)
    plt.ylabel('验证损失')
    plt.title('模型性能对比')
    plt.xticks(rotation=45)
    
    for bar, loss in zip(bars, losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.4f}', ha='center', va='bottom')
    
    # 特征数量对比
    plt.subplot(1, 3, 3)
    feature_counts = [results['simple']['features'], results['medium']['features'], results['advanced']['features']]
    bars = plt.bar(models, feature_counts, color=colors, alpha=0.7)
    plt.ylabel('特征数量')
    plt.title('使用的特征数量')
    plt.xticks(rotation=45)
    
    for bar, count in zip(bars, feature_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 详细结果分析
    print("\n" + "="*80)
    print("实验结果详细分析:")
    print("="*80)
    
    for i, (name, result) in enumerate([('简单基线', results['simple']), 
                                       ('中阶基线', results['medium']), 
                                       ('强基线', results['advanced'])]):
        print(f"\n{i+1}. {name}:")
        print(f"   验证损失: {result['loss']:.6f}")
        print(f"   特征数量: {result['features']}")
        print(f"   训练轮次: {len(result['record']['dev'])}")
    
    # 计算改进幅度
    simple_loss = results['simple']['loss']
    medium_loss = results['medium']['loss']
    advanced_loss = results['advanced']['loss']
    
    medium_improvement = ((simple_loss - medium_loss) / simple_loss) * 100
    advanced_improvement = ((simple_loss - advanced_loss) / simple_loss) * 100
    
    print(f"\n性能改进分析:")
    print(f"中阶基线相对简单基线改进: {medium_improvement:.2f}%")
    print(f"强基线相对简单基线改进: {advanced_improvement:.2f}%")
    print(f"强基线相对中阶基线改进: {((medium_loss - advanced_loss) / medium_loss) * 100:.2f}%")
    
    # 6. 优化总结
    print(f"\n" + "="*80)
    print("优化策略总结:")
    print("="*80)
    print("1. 修复的原代码错误:")
    print("   ✅ 特征选择未实现 (target_only的pass语句)")
    print("   ✅ 标准化位置错误 (使用训练集统计量)")
    print("   ✅ 数据分割方式简单 (改用sklearn.train_test_split)")
    print("   ✅ 缺少L2正则化实现")
    print("   ✅ 没有学习率调度")
    
    print("\n2. 实现的优化策略:")
    print("   🚀 特征工程: 40州 + 2测试阳性 + 症状特征")
    print("   🚀 网络架构: 残差连接 + BatchNorm + Dropout")
    print("   🚀 训练策略: Adam优化器 + 学习率调度 + L2正则化")
    print("   🚀 实验设计: 三级基线对比 + 全面性能分析")
    
    return results

if __name__ == "__main__":
    comprehensive_comparison()
