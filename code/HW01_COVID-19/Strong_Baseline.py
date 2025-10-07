## 导入一些包

# PyTorch - 深度学习框架
import torch  # PyTorch主库
import torch.nn as nn  # 神经网络模块
from sympy.physics.units import femto
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器

# 数据预处理相关
import numpy as np  # 数值计算库
import csv  # CSV文件读取
import os  # 操作系统接口

# 绘图相关
import matplotlib.pyplot as plt  # 绘图库
from matplotlib.pyplot import figure  # 图形创建

# 设置随机种子以确保结果可重现
myseed = 42069  # 设置随机种子
torch.backends.cudnn.deterministic = True  # 确保CUDNN的确定性
torch.backends.cudnn.benchmark = False  # 关闭CUDNN的基准测试模式
np.random.seed(myseed)  # 设置numpy随机种子
torch.manual_seed(myseed)  # 设置PyTorch随机种子
if torch.cuda.is_available():  # 如果GPU可用
    torch.cuda.manual_seed_all(myseed)  # 设置所有GPU的随机种子


## 工具函数部分 - 提供设备选择、绘图等辅助功能
def get_device():
    ''' 获取计算设备（如果GPU可用则使用GPU，否则使用CPU） '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' 绘制深度神经网络的学习曲线（训练和验证损失） '''
    total_steps = len(loss_record['train'])  # 总训练步数
    x_1 = range(total_steps)  # 训练步数的x轴坐标
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]  # 验证步数的x轴坐标（采样）
    figure(figsize=(6, 4))  # 创建6x4英寸的图形
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')  # 绘制训练损失曲线（红色）
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')  # 绘制验证损失曲线（青色）
    plt.ylim(0.0, 5.)  # 设置y轴范围
    plt.xlabel('Training steps')  # x轴标签
    plt.ylabel('MSE loss')  # y轴标签
    plt.title('Learning curve of {}'.format(title))  # 图标题
    plt.legend()  # 显示图例
    plt.show()  # 显示图形


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' 绘制深度神经网络的预测结果散点图 '''
    if preds is None or targets is None:  # 如果没有提供预测值和真实值
        model.eval()  # 设置模型为评估模式
        preds, targets = [], []  # 初始化预测值和真实值列表
        for x, y in dv_set:  # 遍历验证数据集
            x, y = x.to(device), y.to(device)  # 将数据移动到指定设备
            with torch.no_grad():  # 禁用梯度计算
                pred = model(x)  # 前向传播获得预测值
                preds.append(pred.detach().cpu())  # 将预测值移到CPU并添加到列表
                targets.append(y.detach().cpu())  # 将真实值移到CPU并添加到列表
        preds = torch.cat(preds, dim=0).numpy()  # 连接所有预测值并转换为numpy数组
        targets = torch.cat(targets, dim=0).numpy()  # 连接所有真实值并转换为numpy数组

    figure(figsize=(5, 5))  # 创建5x5英寸的正方形图形
    plt.scatter(targets, preds, c='r', alpha=0.5)  # 绘制散点图（红色，半透明）
    plt.plot([-0.2, lim], [-0.2, lim], c='b')  # 绘制理想预测线（蓝色对角线）
    plt.xlim(-0.2, lim)  # 设置x轴范围
    plt.ylim(-0.2, lim)  # 设置y轴范围
    plt.xlabel('ground truth value')  # x轴标签（真实值）
    plt.ylabel('predicted value')  # y轴标签（预测值）
    plt.title('Ground Truth v.s. Prediction')  # 图标题
    plt.show()  # 显示图形

## COVID19数据集类 - 负责读取CSV文件、特征提取、数据划分和归一化
class COVID19Dataset(Dataset):
    ''' COVID19数据集的加载和预处理类 '''

    def __init__(self,
                 path,  # 数据文件路径
                 mode='train',  # 模式：'train'训练、'dev'验证、'test'测试
                 target_only=False, 
                 features_to_exclude=None):  # 是否只使用特定特征
        self.mode = mode  # 保存当前模式

        # 读取数据到numpy数组
        with open(path, 'r') as fp:  # 打开CSV文件
            data = list(csv.reader(fp))  # 读取所有行
            data = np.array(data[1:])[:, 1:].astype(float)  # 跳过标题行和第一列，转换为浮点数

        # --- 这是核心修改部分 ---
        # all_feature_indices = list(range(93)) # 假设总共有93个初始特征
        # if features_to_exclude:
        #     # 如果提供了要排除的特征列表，就从全部特征中将其移除
        #     selected_features = [idx for idx in all_feature_indices if idx not in features_to_exclude]
        # else:
        #     # 如果没有提供，就使用全部特征
        #     selected_features = all_feature_indices

        if not target_only:  # 如果使用所有特征
            feats = list(range(93))  # 使用前93个特征
        else:
            # TODO: 使用40个州的特征 + 2个tested_positive特征（索引57和75）
            feats = list(range(40)) + [57, 75]
            # pass

        if mode == 'test':  # 测试模式
            # 测试数据
            # data: 893 x 93 (40个州 + 第1天(18特征) + 第2天(18特征) + 第3天(17特征))
            data = data[:, feats]  # 选择指定特征
            self.data = torch.FloatTensor(data)  # 转换为PyTorch张量
        else:
            # 训练数据（训练集/验证集）
            # data: 2700 x 94 (40个州 + 第1天(18特征) + 第2天(18特征) + 第3天(18特征))
            target = data[:, -1]  # 最后一列是目标值
            data = data[:, feats]  # 选择指定特征

            # 将训练数据划分为训练集和验证集
            if mode == 'train':  # 训练模式
                indices = [i for i in range(len(data)) if i % 10 != 0]  # 90%数据用于训练
            elif mode == 'dev':  # 验证模式
                indices = [i for i in range(len(data)) if i % 10 == 0]  # 10%数据用于验证

            # 转换数据为PyTorch张量
            self.data = torch.FloatTensor(data[indices])  # 特征数据
            self.target = torch.FloatTensor(target[indices])  # 目标数据

        # 特征归一化（可以移除这部分看看会发生什么）
        # 对第40列之后的特征进行标准化（减去均值除以标准差）
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]  # 保存特征维度

        # 打印数据集信息
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # 返回单个样本
        if self.mode in ['train', 'dev']:  # 训练或验证模式
            # 返回特征和目标值
            return self.data[index], self.target[index]
        else:
            # 测试模式（没有目标值）
            return self.data[index]

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

## 数据加载器函数 - 创建数据集并封装到数据加载器中
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False, features_to_exclude=None):
    ''' 生成数据集，然后放入数据加载器中 '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only, features_to_exclude=features_to_exclude)  # 构建数据集
    dataloader = DataLoader(
        dataset, batch_size,  # 数据集和批次大小
        shuffle=(mode == 'train'), drop_last=False,  # 训练时打乱数据，不丢弃最后不完整的批次
        num_workers=n_jobs, pin_memory=True)  # 工作进程数，内存固定（GPU加速）
    return dataloader  # 返回数据加载器

## 深度神经网络类 - 用于回归的简单全连接网络
class NeuralNet(nn.Module):
    ''' 简单的全连接深度神经网络 '''
    def __init__(self, input_dim):  # 输入维度
        super(NeuralNet, self).__init__()  # 调用父类构造函数

        # 定义神经网络结构
        # TODO: 如何修改这个模型以获得更好的性能？
        self.net = nn.Sequential(  # 顺序容器
            nn.Linear(input_dim, 128),  # 全连接层：输入维度 -> 128
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(0),  # Dropout层
            nn.Linear(128, 64),  # 全连接层：输入维度 -> 64
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(0),  # Dropout层
            nn.Linear(64, 1)  # 全连接层：64 -> 1（回归输出）
        )

        # 均方误差损失函数
        self.criterion = nn.MSELoss(reduction='mean')  # 计算平均MSE损失

    def forward(self, x):
        ''' 给定输入大小(batch_size x input_dim)，计算网络输出 '''
        return self.net(x).squeeze(1)  # 前向传播并压缩最后一维

    def cal_loss(self, pred, target):
        ''' 计算损失 '''
        # TODO: 可以在这里实现L1/L2正则化
        return self.criterion(pred, target)  # 返回预测值和目标值之间的MSE损失

## 训练/验证/测试函数
def train(tr_set, dv_set, model, config, device):
    ''' 深度神经网络训练函数 '''

    n_epochs = config['n_epochs']  # 最大训练轮数

    # 设置优化器
    optimizer = getattr(torch.optim, config['optimizer'])(  # 动态获取优化器类
        model.parameters(), **config['optim_hparas'])  # 传入模型参数和超参数

    # --- 新增代码 ---
    # 定义学习率调度器
    # 当验证集损失连续 15 个 epoch 没有改善时，将当前学习率乘以 0.1
    # patience=15 意味着给模型更多的耐心
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.08, patience=17)

    min_mse = 1000.  # 初始化最小MSE损失
    loss_record = {'train': [], 'dev': []}  # 记录训练和验证损失
    early_stop_cnt = 0  # 早停计数器
    epoch = 0  # 当前轮数
    while epoch < n_epochs:  # 训练循环
        model.train()  # 设置模型为训练模式
        for x, y in tr_set:  # 遍历训练数据加载器
            optimizer.zero_grad()  # 清零梯度
            x, y = x.to(device), y.to(device)  # 将数据移动到指定设备
            pred = model(x)  # 前向传播（计算输出）
            mse_loss = model.cal_loss(pred, y)  # 计算损失
            mse_loss.backward()  # 反向传播（计算梯度）
            optimizer.step()  # 使用优化器更新模型参数
            loss_record['train'].append(mse_loss.detach().cpu().item())  # 记录训练损失

        # 每个epoch后，在验证集上测试模型
        dev_mse = dev(dv_set, model, device)  # 计算验证集MSE

        # --- 新增代码 ---
        # 调度器根据 dev_mse 的表现来决定是否要降低学习率
        # scheduler.step(dev_mse)

        if dev_mse < min_mse:  # 如果模型性能提升
            # 保存改进的模型
            min_mse = dev_mse  # 更新最小MSE
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))  # 打印保存信息
            torch.save(model.state_dict(), config['save_path'])  # 保存模型到指定路径
            early_stop_cnt = 0  # 重置早停计数器
        else:
            early_stop_cnt += 1  # 增加早停计数器

        epoch += 1  # 增加轮数
        loss_record['dev'].append(dev_mse)  # 记录验证损失
        if early_stop_cnt > config['early_stop']:  # 如果连续多轮没有改进
            # 停止训练（早停机制）
            break

    print('Finished training after {} epochs'.format(epoch))  # 打印训练完成信息
    return min_mse, loss_record  # 返回最小MSE和损失记录

def dev(dv_set, model, device):
    ''' 验证函数 - 在验证集上评估模型性能 '''
    model.eval()  # 设置模型为评估模式
    total_loss = 0  # 初始化总损失
    for x, y in dv_set:  # 遍历验证数据加载器
        x, y = x.to(device), y.to(device)  # 将数据移动到指定设备
        with torch.no_grad():  # 禁用梯度计算
            pred = model(x)  # 前向传播（计算输出）
            mse_loss = model.cal_loss(pred, y)  # 计算损失
        total_loss += mse_loss.detach().cpu().item() * len(x)  # 累积损失
    total_loss = total_loss / len(dv_set.dataset)  # 计算平均损失

    return total_loss  # 返回平均损失

def test(tt_set, model, device):
    ''' 测试函数 - 在测试集上进行预测 '''
    model.eval()  # 设置模型为评估模式
    preds = []  # 初始化预测结果列表
    for x in tt_set:  # 遍历测试数据加载器
        x = x.to(device)  # 将数据移动到指定设备
        with torch.no_grad():  # 禁用梯度计算
            pred = model(x)  # 前向传播（计算输出）
            preds.append(pred.detach().cpu())  # 收集预测结果
    preds = torch.cat(preds, dim=0).numpy()  # 连接所有预测结果并转换为numpy数组
    return preds  # 返回预测结果

## 设置超参数和配置
device = get_device()  # 获取当前可用设备（'cpu'或'cuda'）
os.makedirs('models', exist_ok=True)  # 创建models目录，训练的模型将保存到./models/
target_only = False # TODO: 是否只使用40个州和2个tested_positive特征

# TODO: 如何调整这些超参数来提高模型性能？
config = {
    'n_epochs': 3000,  # 最大训练轮数
    'batch_size': 270,  # 数据加载器的小批次大小
    'optimizer': 'Adam',  # 优化算法（torch.optim中的优化器）
    'optim_hparas': {  # 优化器的超参数（取决于使用的优化器）
        'lr': 0.0007,  # SGD的学习率
        # 'momentum': 0.9,  # SGD的动量
        'weight_decay': 7e-6  # SGD的权重衰减
    },
    'early_stop': 200,  # 早停轮数（模型最后一次改进后的轮数）
    'save_path': 'models/model.pth'  # 模型保存路径
}

## 加载数据和创建数据加载器
tr_path = 'data/covid.train.csv'  # 训练数据路径
tt_path = 'data/covid.test.csv'  # 测试数据路径

# 定义要排除的特征索引
day1_features_to_exclude = list(range(40, 58))

# 创建训练、验证和测试数据加载器
tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only, features_to_exclude=day1_features_to_exclude)  # 训练集
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only, features_to_exclude=day1_features_to_exclude)  # 验证集
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only, features_to_exclude=day1_features_to_exclude)  # 测试集

model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

## 开始训练
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
plot_learning_curve(model_loss_record, title='deep model')
del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set

## 测试
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'Medium_outcome/pred.csv')         # save prediction file to pred.csv