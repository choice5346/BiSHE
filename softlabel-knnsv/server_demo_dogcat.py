import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
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
    elif feature_type == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Identity()
        out_dim = 1024
    elif feature_type == 'convnext_tiny':
        model = torchvision.models.convnext_tiny(pretrained=True)
        # ConvNeXt 的 classifier 包含 Flatten，所以我们要保留 Flatten(1)
        model.classifier = nn.Flatten(1)
        out_dim = 768
    elif feature_type == 'vit_b_16':
        model = torchvision.models.vit_b_16(pretrained=True)
        model.heads = nn.Identity()
        out_dim = 768
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
    
    # 🌟 使用 DataParallel 多卡并行!
    if torch.cuda.device_count() > 1:
        print(f"🚀 利用 {torch.cuda.device_count()} 块 GPU 进行特征提取")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    model.eval()

    features_list = []
    # 使用混合精度加速 (fp16)
    scaler = torch.cuda.amp.GradScaler() 
    
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc=f"Extraction"):
            inputs = inputs.to(device, non_blocking=True)
            
            # 开启自动混合精度以加速
            with torch.cuda.amp.autocast():
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
        # 如果 download=True, torchvision 内部会自动检查文件是否存在和完整性 (check_integrity)
        # 只要本地文件完好，它就不会重复下载，而是直接显示 "Files already downloaded and verified"
        full_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        # 尝试 download=False 强行加载
        try:
            full_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform)
        except Exception as e2:
            print(f"❌ 再次尝试加载失败: {e2}")
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

    # --- 重度计算：特征提取 (Multi-GPU/Batch Size 优化) ---
    batch_size = 256 # 🚀 加大 Batch Size 以充分利用服务器算力
    num_workers = 4 # 多线程加载数据
    
    if feature_type != 'raw':
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
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
                        choices=['raw', 'resnet18', 'resnet50', 'vgg11', 'mobilenet_v2', 'densenet121', 
                                 'efficientnet_b0', 'convnext_tiny', 'vit_b_16'],
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
    
    # --- Metric 1: F1-Rank (基于Top-K排序) ---
    # 取全部数据的 flip_ratio 比例作为切分点
    cutoff_k = int(len(sv) * args.flip_ratio)
    # 找到分数最低的 cutoff_k 个数据的阈值
    if cutoff_k > 0:
        threshold_rank = np.sort(sv)[cutoff_k]
        pred_rank = np.zeros(len(sv))
        pred_rank[sv < threshold_rank] = 1
        f1_rank = f1_score(true_dirty_mask, pred_rank)
    else:
        f1_rank = 0.0

    # --- Metric 2: F1-Cluster (基于KMeans聚类) ---
    # 逻辑：用 KMeans 把分数聚成2类，中心较低的那一类作为脏数据
    if len(np.unique(sv)) > 1: # 防止所有分数都一样导致kmeans报错
        X = sv.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
        min_cluster_center = min(kmeans.cluster_centers_.flatten())
        pred_cluster = np.zeros(len(sv))
        # 只要小于这个中心的样本都被视为脏数据 (模仿 helper.py 的逻辑)
        pred_cluster[sv < min_cluster_center] = 1
        f1_cluster = f1_score(true_dirty_mask, pred_cluster)
    else:
        f1_cluster = 0.0

    print(f"Detection Performance (Task: Find {len(dirty_indices)} flipped labels)")
    print(f" -> AUROC      : {auc:.4f}")
    print(f" -> F1-Rank    : {f1_rank:.4f} (Top-{args.flip_ratio:.0%} cutoff)")
    print(f" -> F1-Cluster : {f1_cluster:.4f} (KMeans cutoff)")
    print("-" * 60)
    
    if auc > 0.9:
        print("🎉 检测效果极佳！算法成功识别了大部分标签错误的数据。")
    elif auc > 0.75:
        print("👍 检测效果良好。")
    else:
        print("⚠️ 检测效果一般，可能特征不够强或噪声太难区分。")

    # ==========================================
    # 4. 下游任务应用：数据清洗后的模型重训练
    # ==========================================
    print("\n🏭 [下游任务验证]：清洗数据是否能提升模型性能？")
    print("-" * 60)
    
    def train_and_eval(name, x_tr, y_tr, x_v, y_v):
        """训练一个简单的线性分类器 (Linear Probing) 并评估准确率"""
        if len(x_tr) == 0:
            print(f"   ► [{name:<15}] (Skipped: No data)")
            return 0.0
            
        # 使用 sklearn 的 LogisticRegression 作为分类头
        # 增加 max_iter 防止不收敛
        clf = LogisticRegression(solver='liblinear', C=1.0, max_iter=2000, random_state=42)
        clf.fit(x_tr, y_tr)
        acc = clf.score(x_v, y_v)
        print(f"   ► [{name:<15}] Val Acc: {acc:.2%} (Samples: {len(x_tr)})")
        return acc

    # Case 1: 原始脏数据训练 (Baseline)
    acc_dirty = train_and_eval("Dirty (Full)", x_train, y_train, x_val, y_val)
    
    # Case 2: 随机剔除 (Random Baseline) - 作为对照组
    # 模拟我们不知道哪些是脏的，随便删掉与噪声比例相当的数据
    n_remove = int(len(sv) * args.flip_ratio) # 假设我们知道大概有多少比例的脏数据
    if n_remove > 0:
        random_indices = np.random.choice(len(y_train), len(y_train) - n_remove, replace=False)
        x_random_train = x_train[random_indices]
        y_random_train = y_train[random_indices]
        acc_random = train_and_eval("Random Drop", x_random_train, y_random_train, x_val, y_val)
    else:
        acc_random = acc_dirty

    # Case 3: 智能清洗 (Smart Cleaning by KNN-SV)
    # 策略 A: 剔除 Top-N% 最低分
    # 注意：sv 越低越可能是脏数据，所以我们按照从小到大排序，丢弃前面的
    sorted_indices = np.argsort(sv)
    keep_indices_rank = sorted_indices[n_remove:] # 保留分数靠后的部分（高分部分）
    acc_clean_rank = train_and_eval(f"Clean (Top-{args.flip_ratio:.0%})", x_train[keep_indices_rank], y_train[keep_indices_rank], x_val, y_val)

    # 策略 B: 剔除 KMeans 聚类中的低分簇 (更加自动)
    if len(np.unique(sv)) > 1:
        X_sv = sv.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_sv)
        
        # 找到两个簇的中心
        centers = kmeans.cluster_centers_.flatten()
        # 找到中心较小的那个簇的标签 (认为是脏数据簇)
        dirty_cluster_label = np.argmin(centers)
        
        # 保留那些不属于脏数据簇的样本
        # kmeans.labels_ == dirty_cluster_label -> 脏数据
        # kmeans.labels_ != dirty_cluster_label -> 干净数据
        keep_mask = (kmeans.labels_ != dirty_cluster_label)
        keep_indices_cluster = np.where(keep_mask)[0]
        
        # 如果 KMeans 把所有数据都当成脏的了（异常情况），就不使用
        if len(keep_indices_cluster) < 10:
             # 回退到全数据，或者只用 rank 策略
             acc_clean_cluster = 0.0
        else:
            acc_clean_cluster = train_and_eval("Clean (Auto)", x_train[keep_indices_cluster], y_train[keep_indices_cluster], x_val, y_val)
    else:
        acc_clean_cluster = 0.0

    print("-" * 60)
    best_clean_acc = max(acc_clean_rank, acc_clean_cluster)
    improvement = best_clean_acc - acc_dirty
    
    if improvement > 0.001:
        print(f"✅ 成功验证！通过删除有害数据，模型精度提升了 +{improvement:.2%}")
    else:
        print(f"⚖️ 提升不明显 (Diff: {improvement:+.2%})。可能是模型鲁棒性太强，或者噪声影响有限。")

if __name__ == "__main__":
    main()
