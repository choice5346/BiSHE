import sys
import os

# --- 强制设置 PyTorch 缓存路径到 D 盘项目目录 ---
# 这样模型会下载到 D:\BiSHE\torch_cache，不再占用 C 盘
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['TORCH_HOME'] = os.path.join(PROJECT_ROOT, 'torch_cache')
print(f"🔧 PyTorch 模型缓存路径已设置为: {os.environ['TORCH_HOME']}")

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.cluster import KMeans
import time
import argparse
from tqdm import tqdm

# --- 导入同目录下的核心模块 ---
try:
    from helper import knn_shapley_JW
except ImportError:
    # 备用方案：确保脚本所在目录在 path 中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    try:
        from helper import knn_shapley_JW
    except ImportError:
        print("❌ 错误：无法找到 helper.py。请确保该文件与本脚本在同一目录下。")
        sys.exit(1)

# ==========================================
# 0. 特征提取工具 (Feature Extractor)
# ==========================================
def build_backbone(feature_type: str):
    """根据指定名称构建预训练特征提取器，并移除分类头。"""

    feature_type = feature_type.lower()

    if feature_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Identity()  # 输出 512 维
        out_dim = 512
    elif feature_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Identity()  # 输出 2048 维
        out_dim = 2048
    elif feature_type == 'vgg11':
        model = torchvision.models.vgg11_bn(pretrained=True)
        model.classifier = nn.Identity()  # 输出 25088 维
        out_dim = 25088
    elif feature_type == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Identity()  # 输出 1280 维
        out_dim = 1280
    elif feature_type == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Identity()  # 输出 1024 维
        out_dim = 1024
    
    # --- 新增前沿模型 ---
    elif feature_type == 'efficientnet_b0':
        # Google EfficientNet: 效率之王
        model = torchvision.models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Identity() # 输出 1280 维
        out_dim = 1280
    
    elif feature_type == 'convnext_tiny':
        # Meta ConvNeXt: 现代 CNN 的巅峰 (对标 Transformer)
        model = torchvision.models.convnext_tiny(pretrained=True)
        # ConvNeXt 的 classifier 包含 Flatten，所以我们要保留 Flatten
        model.classifier = nn.Flatten(1) # 输出 768 维
        out_dim = 768
        
    elif feature_type == 'vit_b_16':
        # Vision Transformer: 纯注意力机制
        model = torchvision.models.vit_b_16(pretrained=True)
        model.heads = nn.Identity() # 输出 768 维
        out_dim = 768
        
    else:
        raise ValueError(f"不支持的特征类型: {feature_type}")

    return model, out_dim


def extract_features(data_loader, feature_type: str):
    """使用指定预训练模型提取深度特征。"""

    print(f"🧠 正在使用 {feature_type} 提取深度特征...")

    model, _ = build_backbone(feature_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    features_list = []

    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc=f"Extraction-{feature_type}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            features_list.append(outputs.cpu().numpy())

    return np.concatenate(features_list, axis=0)

# ==========================================
# 1. 模拟数据准备 (Data Preparation)
# ==========================================
def get_dog_cat_data(n_train=200, n_val=100, flip_ratio=0.1, feature_type='raw'):
    print(f"📥 正在准备数据 (训练集: {n_train}, 验证集: {n_val}, 噪声率: {flip_ratio}, 特征: {feature_type})...")
    
    # 本地数据集路径配置
    LOCAL_DATA_ROOT = r'D:/newNLP/else/CATSVSDOGS/data/train_organized'
    
    # 检查路径是否存在
    if not os.path.exists(LOCAL_DATA_ROOT):
        print(f"❌ 错误：找不到数据集路径: {LOCAL_DATA_ROOT}")
        return None

    # 图像预处理
    if feature_type == 'raw':
        # 原始模式：简单调整大小以统一尺寸 (例如 64x64，不然显存可能爆，计算也慢)
        transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.ToTensor()
        ])
    else:
        # 深度特征模式：统一到 224 并使用 ImageNet 归一化
        transform = transforms.Compose([
            transforms.Resize((224, 224)), # 强制调整大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 加载本地数据集 (ImageFolder会自动把 cat/dog 文件夹转为 label 0/1)
    try:
        full_dataset = torchvision.datasets.ImageFolder(root=LOCAL_DATA_ROOT, transform=transform)
        print(f"📂 成功加载本地数据集: {len(full_dataset)} 张图片")
        print(f"   类别映射: {full_dataset.class_to_idx}") # 确认一下 {'cat': 0, 'dog': 1}
    except Exception as e:
        print(f"❌ 本地数据加载失败: {e}")
        return None
    
    # 随机划分训练集和验证集
    np.random.seed(42)
    total_needed = n_train + n_val
    if total_needed > len(full_dataset):
        print(f"⚠️ 警告：请求的数据量 ({total_needed}) 超过了总数据量 ({len(full_dataset)})，将使用所有可用数据。")
        total_needed = len(full_dataset)
        n_train = int(total_needed * 0.8) # 80% 训练
        n_val = total_needed - n_train
        
    all_indices = np.random.choice(len(full_dataset), total_needed, replace=False)
    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:]

    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)

    # 获取标签 (ImageFolder 的 targets 属性包含了所有标签)
    def get_labels_from_full_dataset(full_ds, indices):
        return np.array([full_ds.targets[i] for i in indices])

    y_train = get_labels_from_full_dataset(full_dataset, train_idx)
    y_val = get_labels_from_full_dataset(full_dataset, val_idx)

    # --- 核心：特征准备 ---
    if feature_type != 'raw':
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)
        
        x_train = extract_features(train_loader, feature_type)
        x_val = extract_features(val_loader, feature_type)
    else:
        # 原始模式：手动分批读取避免爆内存
        def get_flattened_data(subset):
            # 将大批量拆分读取
            loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)
            data_list = []
            print(f"   正在读取并展平 {len(subset)} 张原始图片...")
            for imgs, _ in tqdm(loader):
                # Flatten: (B, C, H, W) -> (B, -1)
                data_list.append(imgs.reshape(imgs.shape[0], -1).numpy())
            return np.concatenate(data_list, axis=0)
            
        x_train = get_flattened_data(train_subset)
        x_val = get_flattened_data(val_subset)

    # --- 注入噪声 (Poisoning) ---
    n_flip = int(n_train * flip_ratio)
    dirty_indices = np.random.choice(n_train, n_flip, replace=False)

    print(f"😈 正在注入噪声：反转 {n_flip} 个样本的标签...")
    if len(dirty_indices) > 0:
        y_train[dirty_indices] = 1 - y_train[dirty_indices]

    return x_train, y_train, x_val, y_val, dirty_indices

# ==========================================
# 2. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feature_type',
        type=str,
        default='raw',
        choices=['raw', 'resnet18', 'resnet50', 'vgg11', 'mobilenet_v2', 'densenet121', 
                 'efficientnet_b0', 'convnext_tiny', 'vit_b_16'],
        help="选择使用的特征类型"
    )
    parser.add_argument('--n_train', type=int, default=500, help="训练数据量")
    parser.add_argument('--n_val', type=int, default=100, help="验证数据量")
    args = parser.parse_args()

    print(f"🚀 开始 Soft-label KNN-Shapley 演示 (Cat vs Dog) | 模式: {args.feature_type}")
    print("=" * 50)

    # 1. 获取数据
    data = get_dog_cat_data(
        n_train=args.n_train, 
        n_val=args.n_val, 
        flip_ratio=0.1, 
        feature_type=args.feature_type
    )
    
    if data is None: return
    x_train, y_train, x_val, y_val, dirty_indices = data

    print("\n🔍 开始计算数据价值 (这可能需要几秒钟)...")
    start_time = time.time()
    
    # 2. 调用核心算法 (K=5 是常用值)
    sv = knn_shapley_JW(x_train, y_train, x_val, y_val, K=5)
    
    duration = time.time() - start_time
    print(f"✅ 计算完成！耗时: {duration:.2f} 秒")

    # 3. 评估效果
    print("\n📊 评估结果 (Evaluation):")
    print("-" * 50)

    # 构造 Ground Truth (1表示脏数据/标签反转)
    true_labels = np.zeros(len(y_train))
    true_labels[dirty_indices] = 1

    # --- Metric 1: F1-Rank (基于Top-K排序) ---
    # 模拟 helper.py 中的 kmeans_f1score(cluster=False)
    # 逻辑：取最低分数的 10% 作为预测的最脏数据
    threshold_rank = np.sort(sv)[int(0.1 * len(sv))]
    pred_rank = np.zeros(len(sv))
    pred_rank[sv < threshold_rank] = 1
    f1_rank = f1_score(true_labels, pred_rank)

    # --- Metric 2: F1-Cluster (基于KMeans聚类) ---
    # 模拟 helper.py 中的 kmeans_f1score(cluster=True)
    # 逻辑：用 KMeans 把分数聚成2类，中心较低的那一类作为脏数据
    # 注意：helper.py 的实现是 `val < min_center`，比较严格
    X = sv.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    min_cluster_center = min(kmeans.cluster_centers_.flatten())
    pred_cluster = np.zeros(len(sv))
    pred_cluster[sv < min_cluster_center] = 1
    f1_cluster = f1_score(true_labels, pred_cluster)

    # --- Metric 3: AUROC ---
    # 分数越低越可能是脏数据，所以取负号
    auc = roc_auc_score(true_labels, -sv) if len(dirty_indices) > 0 else 0

    print(f"Dataset Task: Mislabel Detection (Cat vs Dog)")
    print(f"Value Type  : {args.feature_type.upper()} + KNN-SV")
    print(f"Dirty Ratio : {len(dirty_indices)/len(y_train):.1%}")
    print("-" * 50)
    print(f"*** Evaluation Report ***")
    print(f"F1-Rank   : {f1_rank:.3f} (Top-10% cutoff)")
    print(f"F1-Cluster: {f1_cluster:.3f} (KMeans cutoff)")
    print(f"AUROC     : {auc:.3f}")
    print("-" * 50)
    
    if auc > 0.9:
        print("🎉 完美！(Excellent)")
    elif auc > 0.8:
        print("👍 不错！(Good)")
    else:
        print("⚠️ 效果一般 (Average)")

if __name__ == "__main__":
    main()
