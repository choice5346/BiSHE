import sys
import os
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

# --- 设置项目根目录与 helper 导入 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from helper import knn_shapley_JW
except ImportError:
    print("❌ 错误：无法找到 helper.py。请确保该文件与本脚本在同一目录或 PYTHONPATH 中。")
    sys.exit(1)

# --- 设置缓存路径 (可选) ---
# 尽量将数据和模型下载到当前目录下的 data 和 torch_cache 文件夹，方便管理
PROJECT_ROOT = os.path.dirname(current_dir) # 假设上一级是项目根目录
CACHE_DIR = os.path.join(PROJECT_ROOT, 'torch_cache') 
DATA_DIR = os.path.join(PROJECT_ROOT, 'data') # 数据下载目录

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = CACHE_DIR # 设置 PyTorch 模型缓存路径

print(f"🔧 环境配置:\n   - 模型缓存: {CACHE_DIR}\n   - 数据目录: {DATA_DIR}")

# ==========================================
# 0. 特征提取工具 (与原版保持一致)
# ==========================================
def build_backbone(feature_type: str):
    feature_type = feature_type.lower()
    if feature_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Identity()
        out_dim = 512
    elif feature_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        out_dim = 2048
    elif feature_type == 'vgg11':
        model = torchvision.models.vgg11_bn(pretrained=True)
        model.classifier = nn.Identity()
        out_dim = 25088
    elif feature_type == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Identity()
        out_dim = 1280
    elif feature_type == 'efficientnet_b0':
        model = torchvision.models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Identity()
        out_dim = 1280
    else:
        # 默认回退
        print(f"⚠️ 未知模型 {feature_type}, 使用 ResNet18 作为默认")
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Identity()
        out_dim = 512
    return model, out_dim

def extract_features(data_loader, feature_type: str):
    print(f"🧠 正在使用 {feature_type} 提取特征...")
    model, _ = build_backbone(feature_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    features_list = []
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc=f"Extraction"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # L2 归一化，这对 KNN 尤其重要
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
            features_list.append(outputs.cpu().numpy())

    return np.concatenate(features_list, axis=0)

# ==========================================
# 1. 自动下载并筛选数据 (Server Friendly)
# ==========================================
def get_cifar_dog_cat_data(n_train=2000, n_val=500, flip_ratio=0.1, feature_type='resnet18'):
    """
    使用 CIFAR-10 数据集，自动下载并筛选出 'Cat' (3) 和 'Dog' (5) 两个类别。
    """
    print(f"📥 正在准备数据 (基于 CIFAR-10)...")
    
    # 图像预处理
    if feature_type == 'raw':
        # 原始模式：CIFAR本就很小(32x32)，稍微放大一点
        transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.ToTensor()
        ])
    else:
        # 深度特征模式：Upsample 到 224 以适应 ImageNet 预训练模型
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 下载/加载 CIFAR-10
    try:
        # train=True 包含 50000 张图
        full_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    except Exception as e:
        print(f"❌ 数据下载失败: {e}\n请检查网络连接或手动下载 CIFAR-10 到 {DATA_DIR}")
        return None

    # --- 筛选 Cat (3) 和 Dog (5) ---
    # CIFAR-10 classes: 
    # 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 
    # 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
    CAT_IDX = 3
    DOG_IDX = 5
    
    targets = np.array(full_dataset.targets)
    mask = (targets == CAT_IDX) | (targets == DOG_IDX)
    filtered_indices = np.where(mask)[0]
    
    print(f"📂 筛选完成: 找到 {len(filtered_indices)} 张猫狗图片")
    
    # 将标签重映射为 0 (Cat) 和 1 (Dog)
    # 原始 label: 3 -> 0, 5 -> 1
    
    # 随机打乱并划分
    np.random.seed(42)
    np.random.shuffle(filtered_indices)
    
    total_needed = n_train + n_val
    if total_needed > len(filtered_indices):
        print(f"⚠️ 数据不足，最大可用: {len(filtered_indices)}")
        total_needed = len(filtered_indices)
        n_train = int(total_needed * 0.8)
        n_val = total_needed - n_train
    
    train_indices = filtered_indices[:n_train]
    val_indices = filtered_indices[n_train:n_train+n_val]
    
    # 构造 Subset
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # 提取并转换标签
    def get_binary_labels(subset):
        # Subset.dataset 访问原始 dataset，通过 Subset.indices 获取对应标签
        original_labels = np.array([subset.dataset.targets[i] for i in subset.indices])
        binary_labels = np.zeros_like(original_labels)
        binary_labels[original_labels == DOG_IDX] = 1 # Dog = 1
        binary_labels[original_labels == CAT_IDX] = 0 # Cat = 0
        return binary_labels
        
    y_train = get_binary_labels(train_subset)
    y_val = get_binary_labels(val_subset)
    
    print(f"✅ 数据集就绪: 训练集 {len(y_train)} (Cat:{np.sum(y_train==0)}/Dog:{np.sum(y_train==1)}), 验证集 {len(y_val)}")

    # --- 重度计算：特征提取 ---
    batch_size = 64
    if feature_type != 'raw':
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        x_train = extract_features(train_loader, feature_type)
        x_val = extract_features(val_loader, feature_type)
    else:
        # Raw 模式
        def get_flattened(subset):
            loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
            data_list = []
            print("   读取原始像素...")
            for imgs, _ in tqdm(loader):
                data_list.append(imgs.reshape(imgs.shape[0], -1).numpy())
            return np.concatenate(data_list, axis=0)
            
        x_train = get_flattened(train_subset)
        x_val = get_flattened(val_subset)

    # --- 注入噪声 (Poisoning) ---
    n_flip = int(n_train * flip_ratio)
    dirty_indices = np.random.choice(n_train, n_flip, replace=False)

    if n_flip > 0:
        print(f"😈 注入噪声: 反转 {n_flip} 个训练样本的标签...")
        y_train[dirty_indices] = 1 - y_train[dirty_indices]

    return x_train, y_train, x_val, y_val, dirty_indices

# ==========================================
# 2. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_type', type=str, default='resnet18', 
                        choices=['raw', 'resnet18', 'resnet50', 'vgg11', 'mobilenet_v2', 'efficientnet_b0'],
                        help="特征提取器模型")
    parser.add_argument('--n_train', type=int, default=2000, help="训练样本数")
    parser.add_argument('--n_val', type=int, default=500, help="验证样本数")
    parser.add_argument('--flip_ratio', type=float, default=0.1, help="噪声比例")
    parser.add_argument('--K', type=int, default=5, help="KNN 的 K 值")
    
    args = parser.parse_args()

    print(f"\n🚀 [Server Version] Soft-label KNN-Shapley Demo")
    print(f"   Task: Cat vs Dog (from CIFAR-10)\n   Model: {args.feature_type}\n   Noise: {args.flip_ratio:.0%}")
    print("=" * 60)

    # 1. 准备数据
    data = get_cifar_dog_cat_data(
        n_train=args.n_train,
        n_val=args.n_val,
        flip_ratio=args.flip_ratio,
        feature_type=args.feature_type
    )
    if data is None: return
    x_train, y_train, x_val, y_val, dirty_indices = data

    print("\n🔍 正在计算数据价值 (Soft-label KNN-Shapley)...")
    start_time = time.time()
    
    # 2. 核心算法
    # 注意：helper 中的算法是通用的，无需修改，直接传入 numpy 数组即可
    sv = knn_shapley_JW(x_train, y_train, x_val, y_val, K=args.K)
    
    duration = time.time() - start_time
    print(f"✅ 计算完成！耗时: {duration:.2f} 秒")

    # 3. 结果评估
    print("\n📊 评估报告:")
    print("-" * 60)
    
    true_dirty_mask = np.zeros(len(y_train))
    true_dirty_mask[dirty_indices] = 1 # 1 表示是脏数据
    
    if len(dirty_indices) == 0:
        print("没有注入噪声，跳过检测评估。")
        return

    # Metric: AUROC (分数越低越脏，所以取负号)
    # 我们希望脏数据的 Shapley Value 很低（负贡献），所以 -sv 应该很高
    auc = roc_auc_score(true_dirty_mask, -sv)
    
    # Metric: F1 (假设我们要检测出 Top-N% 最差的数据)
    # 取全部数据的 flip_ratio 比例作为切分点
    cutoff_k = int(len(sv) * args.flip_ratio)
    # 找到分数最低的 cutoff_k 个数据的阈值
    threshold = np.partition(sv, cutoff_k)[cutoff_k]
    
    pred_dirty = np.zeros(len(sv))
    pred_dirty[sv <= threshold] = 1
    
    f1 = f1_score(true_dirty_mask, pred_dirty)
    
    print(f"Detection Performance (Task: Find {len(dirty_indices)} flipped labels)")
    print(f" -> AUROC : {auc:.4f} (越高越好, 1.0完美)")
    print(f" -> F1    : {f1:.4f}  (Top-{args.flip_ratio:.0%} cutoff)")
    print("-" * 60)
    
    if auc > 0.9:
        print("🎉 检测效果极佳！算法成功识别了大部分标签错误的数据。")
    elif auc > 0.75:
        print("👍 检测效果良好。")
    else:
        print("⚠️ 检测效果一般，可能特征不够强或噪声太难区分。")

if __name__ == "__main__":
    main()
