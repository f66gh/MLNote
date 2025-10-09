# 导入必要的库
from pickle import TRUE
import numpy as np  # 用于数值计算和数组操作
# 导入PyTorch深度学习框架（优先导入，确保CUDA正确初始化）
import torch
# 导入Dataset类，用于创建自定义数据集
from torch.utils.data import Dataset
# 导入DataLoader类，用于批量加载数据
from torch.utils.data import DataLoader
# 导入神经网络模块
import torch.nn as nn
# 导入垃圾回收模块，用于释放内存
import gc
# 性能监控
import time
import psutil  # 需要安装: pip install psutil
from collections import Counter

# 定义TIMIT数据集类，继承自PyTorch的Dataset类
class TIMITDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        """
        初始化数据集
        X: 输入特征数据（numpy数组）
        y: 标签数据（可选，用于训练集）
        augment: 是否启用数据增强（只对训练集启用）
        """
        self.augment = augment
        # 将numpy数组转换为PyTorch张量，并转为float类型
        self.data = torch.from_numpy(X).float()
        if y is not None:
            # 如果提供了标签，将其转换为整数类型（使用int而不是已弃用的np.int）
            y = y.astype(int)
            # 转换为PyTorch的长整型张量（用于分类任务）
            self.label = torch.LongTensor(y)
        else:
            # 测试集没有标签，设为None
            self.label = None

    def __getitem__(self, idx):
        """
        根据索引获取单个样本
        idx: 样本索引
        返回: 如果有标签则返回(数据, 标签)，否则只返回数据
        """
        data = self.data[idx].clone() # 创建副本以避免修改原始数据
        if self.label is not None:
            # 只在启用增强时进行数据增强
            if self.augment:
                # 频率遮盖: 随机选择f个连续频率通道置0
                freq_mask_param = 4  #  最多遮盖4个通道
                f = int(np.random.uniform(0, freq_mask_param))
                f0 = int(np.random.uniform(0, 39 - f))
                data = data.view(11, 39) # 变形为 11x39
                data[:, f0:f0+f] = 0
                data = data.view(429) # 变回来

                # 时间遮盖: 随机选择t个连续时间帧置0
                time_mask_param = 2 # 最多遮盖2帧
                t = int(np.random.uniform(0, time_mask_param))
                t0 = int(np.random.uniform(0, 11 - t))
                data = data.view(11, 39) # 变形为 11x39
                data[t0:t0+t, :] = 0
                data = data.view(429) # 变回来
            
            # 训练/验证集：返回处理后的特征和对应标签
            return data, self.label[idx]
            
        else:
            # 测试集：只返回特征数据
            return self.data[idx]

    def __len__(self):
        """
        返回数据集的总样本数量
        """
        return len(self.data)

# 定义分类器神经网络类，继承自nn.Module
class Classifier(nn.Module):
    def __init__(self):
        """
        初始化神经网络结构
        """
        # 调用父类的初始化方法
        super(Classifier, self).__init__()
        # 第一层：输入429维特征，输出1024维
        self.layer1 = nn.Linear(429,  2048)
        # 定义Batch Normalization层
        self.bn1 = nn.BatchNorm1d(2048)
        # 第二层：输入1024维，输出512维
        self.layer2 = nn.Linear(2048, 1024)
        # 定义Batch Normalization层
        self.bn2 = nn.BatchNorm1d(1024)
        # 第三层：输入512维，输出128维
        self.layer3 = nn.Linear(1024, 512)
        # 定义Batch Normalization层
        self.bn3 = nn.BatchNorm1d(512)
        # 第四层：输入512维，输出256维
        self.layer4 = nn.Linear(512, 256)
        # 定义Batch Normalization层
        self.bn4 = nn.BatchNorm1d(256)
        # 输出层：输入128维，输出39个类别（音素数量）
        self.out = nn.Linear(256, 39) 

        # 定义激活函数为Sigmoid函数
        self.act_fn = nn.ReLU()
        # 定义Dropout层
        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        """
        前向传播函数，定义数据在网络中的流动过程
        x: 输入数据
        返回: 网络的输出结果
        """
        # 通过第一层线性变换
        x = self.layer1(x)
        # 应用Batch Normalization层
        x = self.bn1(x)
        # 应用ReLU激活函数
        x = self.act_fn(x)
        # 应用Dropout层
        x = self.dropout(x)

        # 通过第二层线性变换
        x = self.layer2(x)
        # 应用Batch Normalization层
        x = self.bn2(x)
        # 应用ReLU激活函数
        x = self.act_fn(x)
        # 应用Dropout层
        x = self.dropout(x)

        # 通过第三层线性变换
        x = self.layer3(x)
        # 应用Batch Normalization层
        x = self.bn3(x)
        # 应用ReLU激活函数
        x = self.act_fn(x)
        # 应用Dropout层
        x = self.dropout(x)

        # 通过第四层线性变换
        x = self.layer4(x)
        # 应用Batch Normalization层
        x = self.bn4(x)
        # 应用ReLU激活函数
        x = self.act_fn(x)
        # 应用Dropout层
        x = self.dropout(x)

        # 通过输出层，得到39个类别的原始分数
        x = self.out(x)
        
        # 返回最终输出
        return x

# 固定随机种子函数，确保实验结果可重现
def same_seeds(seed):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 如果有GPU，设置CUDA的随机种子
        torch.cuda.manual_seed(seed)
        # 为所有GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  
    # 设置NumPy的随机种子
    np.random.seed(seed)  
    # 关闭CUDNN的benchmark模式，确保结果可重现
    torch.backends.cudnn.benchmark = False
    # 启用CUDNN的确定性模式
    torch.backends.cudnn.deterministic = True

# 检查设备函数：优先使用GPU，如果没有则使用CPU
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# 众数滤波后处理函数
def mode_filter_postprocessing(predictions, window_size=5, method='sliding'):
    """
    对预测结果进行众数滤波后处理，提高时序预测的稳定性
    
    Args:
        predictions: 原始预测结果列表或numpy数组
        window_size: 滤波窗口大小，奇数效果更好
        method: 滤波方法
            - 'sliding': 滑动窗口众数滤波
            - 'median': 中位数滤波（适用于数值型）
            - 'weighted': 加权众数滤波（中心权重更大）
    
    Returns:
        filtered_predictions: 滤波后的预测结果
    """
    predictions = np.array(predictions)
    filtered_predictions = predictions.copy()
    n = len(predictions)
    
    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    
    print(f"🔧 应用众数滤波: 窗口大小={window_size}, 方法={method}")
    
    if method == 'sliding':
        # 滑动窗口众数滤波
        for i in range(n):
            # 计算窗口边界
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            # 获取窗口内的预测值
            window_preds = predictions[start:end]
            
            # 计算众数
            counter = Counter(window_preds)
            filtered_predictions[i] = counter.most_common(1)[0][0]
    
    elif method == 'median':
        # 中位数滤波
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            window_preds = predictions[start:end]
            filtered_predictions[i] = np.median(window_preds)
    
    elif method == 'weighted':
        # 加权众数滤波（中心权重更大）
        weights = np.exp(-np.abs(np.arange(-half_window, half_window + 1)) / (half_window + 1))
        
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            window_preds = predictions[start:end]
            
            # 调整权重长度
            current_weights = weights[max(0, half_window - i):half_window + min(n - i, half_window + 1)]
            
            # 加权投票
            vote_dict = {}
            for pred, weight in zip(window_preds, current_weights):
                vote_dict[pred] = vote_dict.get(pred, 0) + weight
            
            # 选择权重最大的预测
            filtered_predictions[i] = max(vote_dict.items(), key=lambda x: x[1])[0]
    
    # 计算改变的预测数量
    changes = np.sum(predictions != filtered_predictions)
    change_rate = changes / n * 100
    
    print(f"📊 滤波统计: 改变了 {changes}/{n} 个预测 ({change_rate:.2f}%)")
    
    return filtered_predictions.astype(int)

# 多级众数滤波
def multi_level_mode_filter(predictions, window_sizes=[3, 5, 7], weights=[0.5, 0.3, 0.2]):
    """
    多级众数滤波，结合不同窗口大小的结果
    
    Args:
        predictions: 原始预测结果
        window_sizes: 不同的窗口大小列表
        weights: 对应的权重列表
    
    Returns:
        filtered_predictions: 多级滤波后的结果
    """
    predictions = np.array(predictions)
    n = len(predictions)
    
    # 存储不同窗口大小的滤波结果
    filtered_results = []
    
    for window_size in window_sizes:
        filtered_result = mode_filter_postprocessing(predictions, window_size, method='sliding')
        filtered_results.append(filtered_result)
    
    # 加权投票决定最终结果
    final_predictions = np.zeros(n, dtype=int)
    
    for i in range(n):
        vote_dict = {}
        for j, filtered_result in enumerate(filtered_results):
            pred = filtered_result[i]
            vote_dict[pred] = vote_dict.get(pred, 0) + weights[j]
        
        # 选择权重最大的预测
        final_predictions[i] = max(vote_dict.items(), key=lambda x: x[1])[0]
    
    return final_predictions

# 自适应众数滤波
def adaptive_mode_filter(predictions, confidence_scores=None, base_window=5, max_window=11):
    """
    自适应众数滤波，根据预测置信度调整窗口大小
    
    Args:
        predictions: 原始预测结果
        confidence_scores: 预测置信度分数（可选）
        base_window: 基础窗口大小
        max_window: 最大窗口大小
    
    Returns:
        filtered_predictions: 自适应滤波后的结果
    """
    predictions = np.array(predictions)
    n = len(predictions)
    filtered_predictions = predictions.copy()
    
    if confidence_scores is None:
        # 如果没有置信度，使用固定窗口
        return mode_filter_postprocessing(predictions, base_window)
    
    confidence_scores = np.array(confidence_scores)
    
    for i in range(n):
        # 根据置信度调整窗口大小
        # 置信度低的地方使用更大的窗口
        confidence = confidence_scores[i]
        window_size = base_window + int((1 - confidence) * (max_window - base_window))
        
        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        window_preds = predictions[start:end]
        
        # 计算众数
        counter = Counter(window_preds)
        filtered_predictions[i] = counter.most_common(1)[0][0]
    
    return filtered_predictions.astype(int)

# 参数调优函数
def optimize_filter_parameters(predictions, validation_labels=None, window_range=(3, 15), step=2):
    """
    自动优化众数滤波的窗口大小参数
    
    Args:
        predictions: 原始预测结果
        validation_labels: 验证集标签（如果有的话）
        window_range: 窗口大小搜索范围
        step: 搜索步长
    
    Returns:
        best_window: 最优窗口大小
        results: 各窗口大小的评估结果
    """
    results = {}
    best_window = window_range[0]
    best_score = 0
    
    print(f"🔍 开始参数优化，搜索范围: {window_range}, 步长: {step}")
    
    for window_size in range(window_range[0], window_range[1] + 1, step):
        if window_size % 2 == 0:  # 确保奇数
            window_size += 1
            
        filtered_preds = mode_filter_postprocessing(predictions, window_size, method='sliding')
        
        if validation_labels is not None:
            # 如果有验证标签，计算准确率
            accuracy = np.mean(filtered_preds == validation_labels)
            score = accuracy
            print(f"   窗口大小 {window_size}: 准确率 = {accuracy:.4f}")
        else:
            # 如果没有验证标签，使用预测稳定性作为评估指标
            # 计算相邻预测的一致性
            consistency = np.mean(filtered_preds[1:] == filtered_preds[:-1])
            # 计算预测变化率（相对于原始预测）
            change_rate = np.mean(filtered_preds != predictions)
            # 综合评分：稳定性高，变化适中
            score = consistency * (1 - abs(change_rate - 0.1))  # 期望10%左右的变化
            print(f"   窗口大小 {window_size}: 一致性 = {consistency:.4f}, 变化率 = {change_rate:.4f}, 评分 = {score:.4f}")
        
        results[window_size] = score
        
        if score > best_score:
            best_score = score
            best_window = window_size
    
    print(f"✅ 最优窗口大小: {best_window} (评分: {best_score:.4f})")
    return best_window, results

# 滤波效果评估函数
def evaluate_filter_performance(original_preds, filtered_preds, confidence_scores=None):
    """
    评估滤波效果
    
    Args:
        original_preds: 原始预测
        filtered_preds: 滤波后预测
        confidence_scores: 置信度分数
    
    Returns:
        evaluation_metrics: 评估指标字典
    """
    original_preds = np.array(original_preds)
    filtered_preds = np.array(filtered_preds)
    
    # 基本统计
    total_samples = len(original_preds)
    changes = np.sum(original_preds != filtered_preds)
    change_rate = changes / total_samples
    
    # 预测稳定性（相邻预测的一致性）
    original_consistency = np.mean(original_preds[1:] == original_preds[:-1])
    filtered_consistency = np.mean(filtered_preds[1:] == filtered_preds[:-1])
    consistency_improvement = filtered_consistency - original_consistency
    
    # 类别分布变化
    from collections import Counter
    original_dist = Counter(original_preds)
    filtered_dist = Counter(filtered_preds)
    
    # 计算分布差异（KL散度的简化版本）
    all_classes = set(original_preds) | set(filtered_preds)
    dist_diff = 0
    for cls in all_classes:
        orig_prob = original_dist.get(cls, 0) / total_samples
        filt_prob = filtered_dist.get(cls, 0) / total_samples
        if orig_prob > 0 and filt_prob > 0:
            dist_diff += abs(orig_prob - filt_prob)
    
    # 置信度相关分析
    confidence_metrics = {}
    if confidence_scores is not None:
        confidence_scores = np.array(confidence_scores)
        
        # 低置信度样本的改变率
        low_conf_mask = confidence_scores < np.percentile(confidence_scores, 25)
        high_conf_mask = confidence_scores > np.percentile(confidence_scores, 75)
        
        low_conf_change_rate = np.mean(original_preds[low_conf_mask] != filtered_preds[low_conf_mask])
        high_conf_change_rate = np.mean(original_preds[high_conf_mask] != filtered_preds[high_conf_mask])
        
        confidence_metrics = {
            'low_confidence_change_rate': low_conf_change_rate,
            'high_confidence_change_rate': high_conf_change_rate,
            'confidence_selectivity': low_conf_change_rate - high_conf_change_rate
        }
    
    metrics = {
        'total_samples': total_samples,
        'changes': changes,
        'change_rate': change_rate,
        'original_consistency': original_consistency,
        'filtered_consistency': filtered_consistency,
        'consistency_improvement': consistency_improvement,
        'distribution_difference': dist_diff,
        **confidence_metrics
    }
    
    return metrics

# 打印评估报告
def print_evaluation_report(metrics):
    """打印详细的评估报告"""
    print("\n📊 众数滤波效果评估报告")
    print("=" * 50)
    print(f"📈 基本统计:")
    print(f"   - 总样本数: {metrics['total_samples']:,}")
    print(f"   - 改变的预测: {metrics['changes']:,} ({metrics['change_rate']:.2%})")
    
    print(f"\n🎯 预测稳定性:")
    print(f"   - 原始一致性: {metrics['original_consistency']:.4f}")
    print(f"   - 滤波后一致性: {metrics['filtered_consistency']:.4f}")
    print(f"   - 一致性提升: {metrics['consistency_improvement']:+.4f}")
    
    print(f"\n📊 类别分布:")
    print(f"   - 分布变化程度: {metrics['distribution_difference']:.4f}")
    
    if 'confidence_selectivity' in metrics:
        print(f"\n🎲 置信度选择性:")
        print(f"   - 低置信度改变率: {metrics['low_confidence_change_rate']:.2%}")
        print(f"   - 高置信度改变率: {metrics['high_confidence_change_rate']:.2%}")
        print(f"   - 选择性指标: {metrics['confidence_selectivity']:+.4f}")
        
        if metrics['confidence_selectivity'] > 0.05:
            print("   ✅ 滤波器能有效识别低置信度预测")
        else:
            print("   ⚠️ 滤波器的置信度选择性较低")
    
    print("=" * 50)

# GPU监控函数
def monitor_system():
    """监控系统资源使用情况"""
    # CPU监控
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    
    print(f"💻 CPU使用率: {cpu_percent:.1f}% ({cpu_count}核)")
    print(f"💾 内存使用: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    
    # GPU监控（如果可用）
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU内存: {gpu_memory:.1f}GB 已用 / {gpu_cached:.1f}GB 缓存 / {gpu_total:.1f}GB 总计")
        
        # GPU利用率（需要nvidia-ml-py）
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            print(f"🔥 GPU利用率: {gpu_util.gpu}% | 显存: {gpu_util.memory}% | 温度: {gpu_temp}°C | 功耗: {power:.1f}W")
        except ImportError:
            print("💡 安装 nvidia-ml-py 获取更详细的GPU信息: pip install nvidia-ml-py")

# Windows多进程保护 - 所有主要执行代码都必须在这个保护块内
if __name__ == '__main__':
    # 设置多进程启动方法（Windows兼容性）
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # 检查GPU可用性（在加载大量数据前先检查）
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")

    # 打印提示信息，告知用户正在加载数据
    print('Loading data ...')

    # 设置数据文件的根目录路径
    data_root='./timit_11/'
    # 加载训练数据：包含音频特征的numpy数组
    train = np.load(data_root + 'train_11.npy')
    # 加载训练标签：对应训练数据的音素类别标签
    train_label = np.load(data_root + 'train_label_11.npy')
    # 加载测试数据：用于最终预测的音频特征数据
    test = np.load(data_root + 'test_11.npy')

    # 打印数据形状信息
    print('Size of training data: {}'.format(train.shape))
    print('Size of testing data: {}'.format(test.shape))

    # 设置验证集比例为20%
    VAL_RATIO = 0.2

    # 计算训练集的样本数量（总数据的80%）
    percent = int(train.shape[0] * (1 - VAL_RATIO))
    # 将原始训练数据按8:2的比例分割为训练集和验证集
    train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
    # 打印训练集的形状
    print('Size of training set: {}'.format(train_x.shape))
    # 打印验证集的形状
    print('Size of validation set: {}'.format(val_x.shape))

    # 优化批处理大小 - 内存稳定，可以增加batch size提升GPU利用率
    BATCH_SIZE = 512  # 从128增加到256，提高GPU利用率

    # 创建训练数据集对象（启用数据增强）
    train_set = TIMITDataset(train_x, train_y, augment=True)
    # 创建验证数据集对象（不启用数据增强）
    val_set = TIMITDataset(val_x, val_y, augment=False)

    # 优化数据加载器性能 - 充分利用14700KF的多核优势
    # 14700KF规格：20核心(8P+12E)，28线程
    # 数据加载是IO密集型任务，可以使用更多worker
    # 根据经验，可以设置为CPU核心数的0.5-1.5倍
    # 对于14700KF的20核心，可以尝试12-24个worker
    num_workers = 20  # 重新启用多进程：内存已优化，可以安全使用4个进程
    # GPU利用率30%说明数据加载是瓶颈，需要多进程加速

    # 先获取设备信息
    device = get_device()
    pin_memory = True if device == 'cuda' else False  # 启用内存锁定加速GPU传输

    print(f"CPU信息: Intel 14700KF (20核28线程)")
    print(f"数据加载优化: 使用 {num_workers} 个worker进程 (多进程加速)")
    print(f"💡 性能策略: 内存优化后重新启用多进程，提升GPU利用率")
    print(f"💡 批处理优化: BATCH_SIZE={BATCH_SIZE}，充分利用GPU计算能力")

    # 创建训练数据加载器，优化性能参数
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True  # 启用持久worker，减少进程重启开销
    )

    # 创建验证数据加载器，优化性能参数
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True  # 启用持久worker，减少进程重启开销
    )

    print(f"批处理大小: {BATCH_SIZE}")
    print(f"数据加载器工作进程: {num_workers}")
    print(f"内存锁定: {pin_memory}")

    # 积极的内存管理 - 立即释放大型数组
    print(f"数据加载完成，开始内存清理...")
    print(f"清理前内存使用: {psutil.virtual_memory().used/1024**3:.1f}GB")

    # 删除原始numpy数组（保留test用于后续推理）
    del train, train_label
    # 删除分割后的数组
    del train_x, train_y, val_x, val_y
    # 强制垃圾回收
    gc.collect()

    print(f"清理后内存使用: {psutil.virtual_memory().used/1024**3:.1f}GB")
    print(f"内存释放: {psutil.virtual_memory().available/1024**3:.1f}GB 可用")

    # 固定随机种子为0，确保实验可重现
    same_seeds(0)

    # 显示当前使用的设备
    print(f'DEVICE: {device}')

    # 训练参数设置
    num_epoch = 50               # 训练轮数：20个epoch
    learning_rate = 0.0002       # 学习率：0.0001

    # 模型检查点保存路径
    model_path = './model.ckpt'

    # 创建模型实例并移动到指定设备（GPU或CPU）
    model = Classifier().to(device)

    # PyTorch 2.x编译优化 - 检查triton可用性
    try:
        if hasattr(torch, 'compile') and device == 'cuda':
            import triton
            print("启用torch.compile优化以补偿单进程性能...")
            model = torch.compile(model, mode='default')
            print("torch.compile优化已启用")
        else:
            print("torch.compile不可用或使用CPU")
    except ImportError:
        print("⚠️ Triton未安装，跳过torch.compile优化")
        print("💡 可运行: pip install triton 来启用编译优化")

    # 定义损失函数：交叉熵损失，适用于多分类任务
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 定义优化器：Adam优化器，用于更新模型参数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # T_max 通常设置为总的训练轮数 num_epoch
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=0)

    # 混合精度训练设置 - 使用新版API避免警告
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    print(f"混合精度训练: {'启用' if scaler else '禁用'}")

    # 开始训练过程
    # 性能监控
    start_time = time.time()

    # 记录最佳验证准确率，用于保存最好的模型
    best_acc = 0.0
    print("\n" + "="*60)
    print("🚀 开始优化训练 - 硬件配置:")
    print(f"   CPU: Intel 14700KF")
    print(f"   GPU: RTX 4070 Ti SUPER") 
    print(f"   批处理大小: {BATCH_SIZE}")
    print(f"   工作进程: {num_workers}")
    print("="*60)
    
    # 开始训练循环，遍历每个epoch
    for epoch in range(num_epoch):
        # 初始化当前epoch的训练准确率
        train_acc = 0.0
        # 初始化当前epoch的训练损失
        train_loss = 0.0
        # 初始化当前epoch的验证准确率
        val_acc = 0.0
        # 初始化当前epoch的验证损失
        val_loss = 0.0

        # 训练阶段
        model.train() # 将模型设置为训练模式（启用dropout、batch norm等）
        # 遍历训练数据加载器中的每个批次
        for i, data in enumerate(train_loader):
            # 获取输入数据和对应标签
            inputs, labels = data
            # 将数据和标签移动到指定设备（GPU或CPU）
            inputs, labels = inputs.to(device), labels.to(device)
            # 清零梯度，防止梯度累积
            optimizer.zero_grad() 
            
            # 使用混合精度训练
            if scaler:
                # 自动混合精度前向传播（使用新版API）
                with torch.amp.autocast('cuda'):
                    # 前向传播：将输入数据传入模型得到预测结果
                    outputs = model(inputs) 
                    # 计算当前批次的损失值
                    batch_loss = criterion(outputs, labels)
                
                # 获取预测类别：找到输出中概率最大的类别索引
                _, train_pred = torch.max(outputs, 1) # 获取概率最高的类别索引
                
                # 缩放损失并反向传播
                scaler.scale(batch_loss).backward()
                # 更新模型参数
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU训练的标准流程
                # 前向传播：将输入数据传入模型得到预测结果
                outputs = model(inputs) 
                # 计算当前批次的损失值
                batch_loss = criterion(outputs, labels)
                # 获取预测类别：找到输出中概率最大的类别索引
                _, train_pred = torch.max(outputs, 1) # 获取概率最高的类别索引
                # 反向传播：计算梯度
                batch_loss.backward() 
                # 更新模型参数
                optimizer.step() 

            # 累加正确预测的样本数量
            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            # 累加训练损失
            train_loss += batch_loss.item()

        # 验证阶段
        if len(val_set) > 0:
            model.eval() # 将模型设置为评估模式（关闭dropout、batch norm等）
            # 使用torch.no_grad()关闭梯度计算，节省内存和计算资源
            with torch.no_grad():
                # 遍历验证数据加载器中的每个批次
                for i, data in enumerate(val_loader):
                    # 获取验证数据和标签
                    inputs, labels = data
                    # 将数据移动到指定设备
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # 验证阶段也使用混合精度
                    if scaler:
                        with torch.amp.autocast('cuda'):
                            # 前向传播得到预测结果
                            outputs = model(inputs)
                            # 计算验证损失
                            batch_loss = criterion(outputs, labels)
                    else:
                        # 前向传播得到预测结果
                        outputs = model(inputs)
                        # 计算验证损失
                        batch_loss = criterion(outputs, labels)
                    
                    # 获取预测类别
                    _, val_pred = torch.max(outputs, 1) 
                
                    # 累加正确预测的验证样本数量
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # 获取概率最高的类别索引
                    # 累加验证损失
                    val_loss += batch_loss.item()

                # 打印当前epoch的训练和验证结果
                elapsed_time = time.time() - start_time
                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f} | 时间: {:.1f}s'.format(
                    epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader), elapsed_time
                ))
                
                # 每5个epoch监控一次系统资源
                if (epoch + 1) % 5 == 0:
                    print("📊 系统资源监控:")
                    monitor_system()
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("🧹 GPU缓存已清理")
                    print("-" * 50)

                # 如果当前模型性能更好，保存模型检查点
                if val_acc > best_acc:
                    best_acc = val_acc
                    # 保存模型的状态字典（参数）
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
        else:
            # 如果没有验证集，只打印训练结果
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
            ))
        # scheduler.step()

    # 如果没有进行验证，保存最后一个epoch的模型
    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')

    # 创建测试数据集
    test_set = TIMITDataset(test, None)  # 测试集没有标签，传入None
    # 创建测试数据加载器，使用保守参数
    test_loader = DataLoader(
        test_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=16,  # 测试阶段使用较少的worker
        pin_memory=pin_memory,
        persistent_workers=False
    )

    # 创建新的模型实例并加载训练好的权重
    model = Classifier().to(device)
    # 从检查点文件加载模型参数
    model.load_state_dict(torch.load(model_path))

    # 为推理阶段也启用编译优化
    try:
        if hasattr(torch, 'compile') and device == 'cuda':
            import triton
            print("为推理阶段启用torch.compile...")
            model = torch.compile(model, mode='default')
            print("推理优化已启用")
    except ImportError:
        print("推理阶段跳过torch.compile（Triton未安装）")

    # 初始化预测结果列表和置信度列表
    predict = []
    confidence_scores = []
    model.eval() # 将模型设置为评估模式
    # 关闭梯度计算，进行推理
    with torch.no_grad():
        # 遍历测试数据的每个批次
        for i, data in enumerate(test_loader):
            # 获取测试输入数据
            inputs = data
            # 将数据移动到指定设备
            inputs = inputs.to(device)
            
            # 推理阶段也使用混合精度
            if scaler:
                with torch.amp.autocast('cuda'):
                    # 前向传播得到预测输出
                    outputs = model(inputs)
            else:
                # 前向传播得到预测输出
                outputs = model(inputs)
            
            # 计算softmax概率分布
            probabilities = torch.softmax(outputs, dim=1)
            # 获取预测类别：找到概率最大的类别索引
            _, test_pred = torch.max(outputs, 1) # 获取概率最高的类别索引
            # 获取最大概率作为置信度
            max_probs, _ = torch.max(probabilities, dim=1)

            # 将预测结果转换为numpy数组并添加到预测列表中
            for y in test_pred.cpu().numpy():
                predict.append(y)
            
            # 保存置信度分数
            for conf in max_probs.cpu().numpy():
                confidence_scores.append(conf)

    print(f"\n🎯 原始预测完成，共 {len(predict)} 个样本")
    print(f"📊 平均置信度: {np.mean(confidence_scores):.4f}")
    
    # 先在验证集上测试后处理效果
    print("\n" + "="*60)
    print("🧪 在验证集上测试后处理效果...")
    
    # 对验证集进行预测以评估后处理效果
    val_predictions = []
    val_confidences = []
    val_true_labels = []
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            probabilities = torch.softmax(outputs, dim=1)
            _, val_pred = torch.max(outputs, 1)
            max_probs, _ = torch.max(probabilities, dim=1)
            
            val_predictions.extend(val_pred.cpu().numpy())
            val_confidences.extend(max_probs.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())
    
    # 计算原始验证准确率
    original_val_acc = np.mean(np.array(val_predictions) == np.array(val_true_labels))
    print(f"📊 验证集原始准确率: {original_val_acc:.6f} ({original_val_acc*100:.4f}%)")
    
    # 在验证集上应用后处理并计算准确率提升
    print("\n🔧 在验证集上测试不同滤波方法...")
    
    # 测试不同窗口大小
    best_val_acc = original_val_acc
    best_window_val = 5
    
    for window_size in [3, 5, 7, 9, 11]:
        filtered_val_preds = mode_filter_postprocessing(val_predictions, window_size, method='sliding')
        filtered_val_acc = np.mean(np.array(filtered_val_preds) == np.array(val_true_labels))
        improvement = filtered_val_acc - original_val_acc
        
        print(f"   窗口={window_size}: {filtered_val_acc:.6f} ({filtered_val_acc*100:.4f}%) 提升: {improvement:+.6f} ({improvement*100:+.4f}%)")
        
        if filtered_val_acc > best_val_acc:
            best_val_acc = filtered_val_acc
            best_window_val = window_size
    
    print(f"\n✅ 验证集最佳窗口大小: {best_window_val}")
    print(f"📈 验证集最大准确率提升: {(best_val_acc - original_val_acc)*100:+.4f}%")
    
    # 应用众数滤波后处理
    print("\n" + "="*60)
    print("🔧 开始应用众数滤波后处理...")
    
    # 步骤1: 使用验证集确定的最佳参数
    print(f"\n🔍 第一步：使用验证集优化的参数 (窗口大小={best_window_val})")
    optimal_window = best_window_val
    
    # 步骤2: 应用不同的滤波方法
    print(f"\n🛠️ 第二步：应用多种滤波方法")
    
    # 方法1: 优化后的基础滑动窗口众数滤波
    print(f"\n1️⃣ 优化的滑动窗口众数滤波 (窗口={optimal_window})")
    filtered_predict_optimized = mode_filter_postprocessing(predict, window_size=optimal_window, method='sliding')
    
    # 方法2: 加权众数滤波
    print(f"\n2️⃣ 加权众数滤波 (窗口={optimal_window+2})")
    filtered_predict_weighted = mode_filter_postprocessing(predict, window_size=optimal_window+2, method='weighted')
    
    # 方法3: 多级众数滤波
    print(f"\n3️⃣ 多级众数滤波 (窗口=[{optimal_window-2},{optimal_window},{optimal_window+2}])")
    multi_windows = [max(3, optimal_window-2), optimal_window, optimal_window+2]
    filtered_predict_multi = multi_level_mode_filter(predict, window_sizes=multi_windows, weights=[0.5, 0.3, 0.2])
    
    # 方法4: 自适应众数滤波（基于置信度）
    print(f"\n4️⃣ 自适应众数滤波 (基于置信度)")
    filtered_predict_adaptive = adaptive_mode_filter(predict, confidence_scores, base_window=optimal_window-2, max_window=optimal_window+4)
    
    # 步骤3: 评估各种方法的效果
    print(f"\n📊 第三步：评估各种滤波方法的效果")
    
    methods = {
        '优化滑动窗口': filtered_predict_optimized,
        '加权众数滤波': filtered_predict_weighted, 
        '多级众数滤波': filtered_predict_multi,
        '自适应滤波': filtered_predict_adaptive
    }
    
    method_scores = {}
    for method_name, filtered_preds in methods.items():
        print(f"\n--- {method_name} ---")
        metrics = evaluate_filter_performance(predict, filtered_preds, confidence_scores)
        print_evaluation_report(metrics)
        
        # 计算综合评分
        score = (metrics['consistency_improvement'] * 0.4 + 
                (1 - metrics['change_rate']) * 0.3 + 
                metrics.get('confidence_selectivity', 0) * 0.3)
        method_scores[method_name] = score
        print(f"🏆 综合评分: {score:.4f}")
    
    # 选择最佳方法
    best_method = max(method_scores.items(), key=lambda x: x[1])
    final_predictions = methods[best_method[0]]
    
    print(f"\n" + "="*60)
    print(f"🏆 最佳滤波方法: {best_method[0]} (评分: {best_method[1]:.4f})")
    print(f"📈 最终预测变化统计:")
    changes = np.sum(np.array(predict) != np.array(final_predictions))
    print(f"   - 改变的预测数量: {changes}/{len(predict)} ({changes/len(predict)*100:.2f}%)")
    
    # 基于验证集结果预估测试集准确率提升
    val_improvement = (best_val_acc - original_val_acc) * 100
    print(f"\n🎯 预期效果 (基于验证集结果):")
    print(f"   - 验证集准确率提升: {val_improvement:+.4f}%")
    print(f"   - 预期测试集也会有类似的准确率提升")
    if val_improvement > 0:
        print(f"   ✅ 后处理预期能提升模型性能")
    else:
        print(f"   ⚠️ 后处理可能不会显著提升性能，建议使用原始预测")
    
    # 保存原始预测结果
    with open('prediction_raw.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(predict):
            f.write('{},{}\n'.format(i, y))
    
    # 保存滤波后的预测结果
    with open('prediction.csv', 'w') as f:
        # 写入CSV文件头
        f.write('Id,Class\n')
        # 写入每个样本的预测结果
        for i, y in enumerate(final_predictions):
            f.write('{},{}\n'.format(i, y))
    
    # 保存所有滤波方法的结果用于比较
    with open('prediction_comparison.csv', 'w') as f:
        f.write('Id,Raw,Optimized_Filter,Weighted_Filter,Multi_Filter,Adaptive_Filter\n')
        for i in range(len(predict)):
            f.write('{},{},{},{},{},{}\n'.format(
                i, 
                predict[i], 
                filtered_predict_optimized[i],
                filtered_predict_weighted[i], 
                filtered_predict_multi[i],
                filtered_predict_adaptive[i]
            ))

    print("\n🎉 训练和推理完成！")
    print("📁 输出文件说明:")
    print("   - prediction.csv: 最终的滤波后预测结果")
    print("   - prediction_raw.csv: 原始预测结果（未滤波）")
    print("   - prediction_comparison.csv: 各种滤波方法的对比结果")
    
    print(f"\n📊 性能总结:")
    print(f"   - 验证集原始准确率: {original_val_acc*100:.4f}%")
    print(f"   - 验证集最佳滤波准确率: {best_val_acc*100:.4f}%")
    print(f"   - 准确率提升: {val_improvement:+.4f}%")
    print(f"   - 最佳滤波方法: {best_method[0]}")
    print(f"   - 最佳窗口大小: {best_window_val}")
    
    print("\n💡 使用建议:")
    if val_improvement > 0.1:  # 提升超过0.1%
        print("   ✅ 强烈推荐使用 prediction.csv (滤波后结果)")
        print(f"   📈 预期测试集准确率提升约 {val_improvement:.4f}%")
    elif val_improvement > 0:
        print("   ✅ 建议使用 prediction.csv (有轻微提升)")
        print(f"   📈 预期测试集准确率提升约 {val_improvement:.4f}%")
    else:
        print("   ⚠️ 建议使用 prediction_raw.csv (滤波效果不明显)")
        print("   📉 滤波可能不会提升此数据集的性能")
    
    print("\n🔧 进一步优化建议:")
    print("   - 可以尝试调整多级滤波的权重组合")
    print("   - 考虑使用更复杂的时序模型(如LSTM/Transformer)")
    print("   - 分析预测错误的模式，针对性改进模型结构")
    
    print(f"\n🏁 任务完成! 众数滤波后处理已成功集成到音素分类流程中。")