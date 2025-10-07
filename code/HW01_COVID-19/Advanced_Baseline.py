## 高级优化模型 - 基于Strong_Baseline的进一步改进
## 目标：将损失从0.7057进一步降低

# 导入所有必要的库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch.nn.functional as F

# 设置随机种子
myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

## 工具函数
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
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

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

## 改进的数据集类 - 添加数据增强和更好的特征工程
class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False, use_feature_engineering=True):
        self.mode = mode
        self.use_feature_engineering = use_feature_engineering

        # 读取数据
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            # 优化特征选择：使用更多相关特征
            feats = list(range(40)) + [57, 75]  # 州特征 + tested_positive特征

        if mode == 'test':
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats]

            # 数据划分
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # 特征工程和归一化
        if self.use_feature_engineering:
            self.apply_feature_engineering()
        
        # 标准归一化
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / (self.data[:, 40:].std(dim=0, keepdim=True) + 1e-8)  # 添加小常数防止除零

        self.dim = self.data.shape[1]
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def apply_feature_engineering(self):
        """应用特征工程技术"""
        if self.data.shape[1] >= 93:  # 确保有足够的特征
            # 创建交互特征（前几个重要特征的乘积）
            interaction_features = []
            
            # 选择一些重要特征进行交互
            important_indices = [57, 75] if self.data.shape[1] > 75 else [40, 41]
            
            for i in range(len(important_indices)):
                for j in range(i+1, len(important_indices)):
                    if important_indices[i] < self.data.shape[1] and important_indices[j] < self.data.shape[1]:
                        interaction = (self.data[:, important_indices[i]] * self.data[:, important_indices[j]]).unsqueeze(1)
                        interaction_features.append(interaction)
            
            if interaction_features:
                self.data = torch.cat([self.data] + interaction_features, dim=1)

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False, use_feature_engineering=True):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only, 
                            use_feature_engineering=use_feature_engineering)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader

## 改进的神经网络 - 使用更先进的架构
class AdvancedNeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(AdvancedNeuralNet, self).__init__()
        
        # 使用更深的网络和残差连接
        self.input_layer = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # 第一个残差块
        self.hidden1 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.hidden2 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        # 第二个残差块
        self.hidden3 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.hidden4 = nn.Linear(128, 128)
        self.bn5 = nn.BatchNorm1d(128)
        
        # 输出层
        self.hidden5 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # 损失函数
        self.criterion = nn.MSELoss(reduction='mean')
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """改进的权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 输入层
        x = F.relu(self.bn1(self.input_layer(x)))
        x = self.dropout(x)
        
        # 第一个残差块
        residual = x
        x = F.relu(self.bn2(self.hidden1(x)))
        x = self.dropout(x)
        x = self.bn3(self.hidden2(x))
        x = F.relu(x + residual)  # 残差连接
        
        # 降维
        x = F.relu(self.bn4(self.hidden3(x)))
        x = self.dropout(x)
        
        # 第二个残差块
        residual = x
        x = self.bn5(self.hidden4(x))
        x = F.relu(x + residual)  # 残差连接
        
        # 输出层
        x = F.relu(self.bn6(self.hidden5(x)))
        x = self.dropout(x)
        x = self.output(x)
        
        return x.squeeze(1)
    
    def cal_loss(self, pred, target):
        """计算损失，可以添加正则化"""
        mse_loss = self.criterion(pred, target)
        
        # 添加L1正则化
        l1_reg = 0
        for param in self.parameters():
            l1_reg += torch.sum(torch.abs(param))
        
        # 总损失 = MSE + L1正则化
        total_loss = mse_loss + 1e-6 * l1_reg
        return total_loss

## 改进的训练函数
def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    
    # 优化器
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    
    # 学习率调度器 - 余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-7)
    
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    
    while epoch < n_epochs:
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += mse_loss.detach().cpu().item()
            num_batches += 1
            loss_record['train'].append(mse_loss.detach().cpu().item())
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        dev_mse = dev(dv_set, model, device)
        
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f}, lr = {:.2e})'
                .format(epoch + 1, min_mse, scheduler.get_last_lr()[0]))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        
        if early_stop_cnt > config['early_stop']:
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.criterion(pred, y)  # 验证时只用MSE，不用正则化
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)
    return total_loss

def test(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

## 配置
device = get_device()
os.makedirs('models', exist_ok=True)

# 改进的配置
config = {
    'n_epochs': 5000,  # 增加训练轮数
    'batch_size': 128,  # 减小批次大小，增加更新频率
    'optimizer': 'AdamW',  # 使用AdamW优化器
    'optim_hparas': {
        'lr': 0.001,  # 稍微提高初始学习率，让调度器来控制
        'weight_decay': 1e-4,  # 适中的权重衰减
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },
    'early_stop': 300,  # 增加早停耐心
    'save_path': 'models/advanced_model.pth'
}

## 数据加载
tr_path = 'data/covid.train.csv'
tt_path = 'data/covid.test.csv'

# 使用特征工程
target_only = False  # 先用所有特征试试
tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], 
                        target_only=target_only, use_feature_engineering=True)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], 
                        target_only=target_only, use_feature_engineering=True)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], 
                        target_only=target_only, use_feature_engineering=True)

## 训练
model = AdvancedNeuralNet(tr_set.dataset.dim).to(device)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
plot_learning_curve(model_loss_record, title='Advanced Model')

# 加载最佳模型
del model
model = AdvancedNeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)

## 预测和保存
def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)
save_pred(preds, 'Advanced_outcome/pred.csv')

print(f"最终验证损失: {model_loss:.4f}")
