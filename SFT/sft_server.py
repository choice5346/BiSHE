import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import shutil
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# 0. 环境与路径配置
# ==========================================

# 设置 HF 镜像 (针对国内网络)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# [Fix OOM] 启用内存碎片优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 定义本地资源保存路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 资源保存在当前脚本同级目录下的 server_resources
RESOURCES_DIR = os.path.join(CURRENT_DIR, "server_resources")
# 这里直接读取 alpaca_data.json 文件
DATASET_PATH = os.path.join(RESOURCES_DIR, "alpaca_data.json")
DATASETS_CONFIG_PATH = os.path.join(CURRENT_DIR, "datasets_config.json")
MODEL_PATH = os.path.join(RESOURCES_DIR, "qwen_model")

# 确保目录存在
os.makedirs(RESOURCES_DIR, exist_ok=True)

CONFIG = {
    # 自动识别：如果本地 MODEL_PATH 里有文件，就用本地路径；否则用云端ID去下载
    "model_path": MODEL_PATH, 

    "model_id_hf": "Qwen/Qwen1.5-0.5B",   # HuggingFace ID
    "model_id_ms": "qwen/Qwen1.5-0.5B",   # ModelScope ID (备用)
    
    # 实验参数
    "n_samples": 1000,                    # 本次实验使用的样本数
    "n_val_samples": 20,                  # 验证集大小
    "poison_ratio": 0.3,                  # 投毒比例
    "output_dir": os.path.join(CURRENT_DIR, "server_results"),
    "seed": 42
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_registry():
    """
    读取数据集注册表，返回 {dataset_name: relative_or_abs_path}
    """
    if not os.path.exists(DATASETS_CONFIG_PATH):
        return {"alpaca_local": DATASET_PATH}

    try:
        with open(DATASETS_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"⚠️ 读取 datasets_config.json 失败，回退默认数据集: {e}")
        return {"alpaca_local": DATASET_PATH}

    registry = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            p = v.get("path", "")
        else:
            p = str(v)

        if not p:
            continue

        if os.path.isabs(p):
            registry[k] = p
        else:
            registry[k] = os.path.join(CURRENT_DIR, p)

    if "alpaca_local" not in registry:
        registry["alpaca_local"] = DATASET_PATH
    return registry


def resolve_dataset_path(dataset_name=None, dataset_path=None):
    """
    优先级: 显式 dataset_path > dataset_name(注册表) > 默认 alpaca_local
    """
    if dataset_path:
        return dataset_path, "custom_path"

    registry = load_dataset_registry()
    if dataset_name and dataset_name in registry:
        return registry[dataset_name], dataset_name

    return registry.get("alpaca_local", DATASET_PATH), "alpaca_local"


def parse_args():
    parser = argparse.ArgumentParser(description="SFT multi-model comparison server")
    parser.add_argument("--dataset", type=str, default="alpaca_local", help="数据集名称（来自 datasets_config.json）")
    parser.add_argument("--dataset_path", type=str, default=None, help="直接指定 JSON 数据路径（优先级高于 --dataset）")
    parser.add_argument("--n_samples", type=int, default=None, help="覆盖默认样本数")
    parser.add_argument("--poison_ratio", type=float, default=None, help="覆盖默认投毒比例")
    parser.add_argument("--n_val_samples", type=int, default=None, help="覆盖默认验证样本数")
    parser.add_argument("--seed", type=int, default=None, help="覆盖默认随机种子")
    parser.add_argument("--output_subdir", type=str, default=None, help="结果输出子目录（例如 dolly_15k）")
    return parser.parse_args()

# ==========================================
# 1. 资源下载与准备 (纯净版 - 不依赖 datasets 库)
# ==========================================

def get_local_model_path():
    """
    检查本地是否有模型，没有则下载 (优先尝试 ModelScope，其次 HuggingFace)
    """
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print(f"✅ 检测到本地模型已存在: {MODEL_PATH}")
        return MODEL_PATH
    
    print(f"📥 本地未找到模型，开始下载...")
    
    # 尝试 ModelScope
    try:
        print("🚀 尝试使用 ModelScope 下载...")
        from modelscope import snapshot_download
        mw_path = snapshot_download(CONFIG['model_id_ms'])
        
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
        shutil.copytree(mw_path, MODEL_PATH)
        print(f"✅ ModelScope 下载完成: {MODEL_PATH}")
        return MODEL_PATH
    except ImportError:
        print("⚠️ 未安装 modelscope, 跳过。")
    except Exception as e:
        print(f"❌ ModelScope 下载失败: {e}")

    # 尝试 HuggingFace
    try:
        print("☁️ 尝试使用 HuggingFace 下载...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=CONFIG['model_id_hf'], local_dir=MODEL_PATH)
        return MODEL_PATH
    except Exception as e:
        print(f"❌ HuggingFace 下载失败: {e}")
        raise RuntimeError("无法下载模型，请手动下载 Qwen1.5-0.5B 到 server_resources/qwen_model")

def prepare_data_local(dataset_path=None):
    """
    加载数据并进行切分、投毒
    返回:
    1. final_data (投毒后的训练集 -> 对应 'Dirty Model')
    2. pure_data (未投毒的纯净训练集 -> 对应 'Oracle Model')
    3. dirty_indices_gt (投毒索引)
    4. oracle_data (用于 Shapley 计算的验证集)
    """
    print("📥 正在读取数据文件...")
    target_dataset_path = dataset_path or DATASET_PATH
    
    ds_full = None
    if os.path.exists(target_dataset_path):
        try:
            with open(target_dataset_path, 'r', encoding='utf-8') as f:
                ds_full = json.load(f)
            print(f"✅ 成功加载 JSON 数据: {len(ds_full)} 条")
            print(f"📄 数据路径: {target_dataset_path}")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
    
    if ds_full is None:
        print("☢️ 未找到数据或加载失败，生成合成数据兜底...")
        ds_full = [{"instruction": f"Solve {k}+{k}", "input":"", "output":f"{k+k}"} for k in range(5000)]

    # 切分前 n_samples
    current_n = CONFIG['n_samples']
    print(f"✂️ 截取前 {current_n} 条数据...")
    
    ds_list = []
    count = 0
    for item in ds_full:
        ds_list.append({
            "instruction": item.get("instruction", ""), 
            "input": item.get("input", ""), 
            "output": item.get("output", "")
        })
        count += 1
        if count >= current_n: break
    
    # 这是一个没有投毒的纯净备份，用来训练 Oracle 模型
    pure_data = [x.copy() for x in ds_list] 
    
    # 这里的 oracle_data 仅用于计算 Shapley 时的“基准”，不参与训练
    # 按照惯例，我们从干净数据里留出一小部分作为验证
    oracle_data = [x.copy() for x in ds_list[:100]] 
    
    # 开始投毒构造 final_data
    final_data = []
    dirty_indices_gt = [] 
    
    set_seed(CONFIG['seed'])
    garbage_responses = ["I don't know.", "Error 404.", "Noise.", "Ignore."]
    
    print(f"😈 注入噪声 ({CONFIG['poison_ratio']:.0%})...")
    for i, item in enumerate(ds_list):
        is_poison = random.random() < CONFIG['poison_ratio']
        new_item = item.copy()
        if is_poison:
            new_item["output"] = random.choice(garbage_responses)
            dirty_indices_gt.append(i)
        final_data.append(new_item)
    
    print(f"✅ 数据准备完毕!")
    print(f"   - Dirty (训练用): {len(final_data)} 条 (投毒 {len(dirty_indices_gt)})")
    print(f"   - Pure (对比用):  {len(pure_data)} 条")
    
    return final_data, pure_data, dirty_indices_gt, oracle_data

# ==========================================
# 2. 梯度与 KNN-Shapley
# ==========================================

# 导入正统的 KNN-Shapley (Jia-Wei) 算法
try:
    # 优先尝试导入同目录下的 helper.py
    # 注意：现在 sft_server.py 和 helper.py 都在 SFT/ 目录下
    from helper import knn_shapley_JW
    print("✅ 成功导入 KNN-Shapley 算法 (from local helper.py)")
except ImportError:
    try:
        # 回退尝试从上级目录导入 (兼容旧逻辑)
        sys.path.append(os.path.join(CURRENT_DIR, "..", "softlabel-knnsv"))
        from helper import knn_shapley
        knn_shapley_JW = knn_shapley
        print("✅ 成功导入 KNN-Shapley 算法 (from ../softlabel-knnsv/helper.py)")
    except ImportError:
        print("❌ 无法导入 knn_shapley 算法，请检查 helper.py 是否存在。")
        # 定义一个简单的 Fallback 防止程序崩溃
        def knn_shapley_JW(train_embeds, test_embeds, y_train, y_test, K=10, **kwargs):
             print("⚠️ 使用极其简化的 Fallback KNN...")
             return np.random.rand(len(train_embeds))
    def knn_shapley_JW(train_embeds, test_embeds, y_train, y_test, K=10, **kwargs):
        # 这是一个兼容接口的简易版，防止找不到文件报错
        # 注意: helper.py 里的接口通常是 (train_X, test_X, train_y, test_y, K)
        # 这里我们没有 y (label)，或者说 y 可以视为 dummy
        N_train = train_embeds.shape[0]
        N_val = test_embeds.shape[0]
        train_grads = F.normalize(torch.tensor(train_embeds), p=2, dim=1)
        val_grads = F.normalize(torch.tensor(test_embeds), p=2, dim=1)
        S = torch.matmul(val_grads, train_grads.T).numpy()
        min_vals = np.zeros(N_train)
        for j in range(N_val):
            s_row = S[j]
            topk_indices = np.argsort(s_row)[-K:]
            min_vals[topk_indices] += s_row[topk_indices]
        return min_vals

def compute_knn_shapley_gradient(train_grads, val_grads, K=10):
    """
    计算 KNN-Shapley 值 (Wrapper)
    统一调用 knn_shapley_JW
    """
    print(f"   -> 调用 KNN-Shapley JW 算法 (K={K})...")
    
    # 确保输入是 numpy，且 float32 防止溢出
    if isinstance(train_grads, torch.Tensor):
        train_np = train_grads.cpu().float().numpy()
    else:
        train_np = train_grads
        
    if isinstance(val_grads, torch.Tensor):
        val_np = val_grads.cpu().float().numpy()
    else:
        val_np = val_grads
        
    # knn_shapley_JW 需要 label y，但我们是做 SFT 数据清洗 (Regression/Generation)
    # 并没有显式的 class label。
    # Hack: 对于 SFT 任务，我们假设所有样本都在同一个“任务空间”下
    # 所以给所有人赋予一个伪标签 y=0，强行让算法只利用 X 的距离 (Similarity)
    # helper.py 里的算法主要依赖 X 的距离，y 用于计算 match (Utility)
    # 如果 y 全一样，Utility 就变成了单纯的相似度衰减，这符合我们的需求
    
    N_train = train_np.shape[0]
    N_val = val_np.shape[0]
    y_train = np.zeros(N_train, dtype=int)
    y_val = np.zeros(N_val, dtype=int)
    
    # helper.py 里的函数签名是: knn_shapley_JW(train_X, test_X, train_y, test_y, K)
    # 返回的是 (N_train,) 的 value
    shapley_values = knn_shapley_JW(train_np, val_np, y_train, y_val, K=K)
    
    return shapley_values

def extract_gradient_features(model_path, dataset_list, indices):
    print(f"🧬 提取梯度... 样本数: {len(indices)}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, peft_config)
    model.train()
    
    grads = []
    subset = [dataset_list[i] for i in indices]
    MAX_LEN = 256
    
    for item in tqdm(subset, desc="Grads"):
        text = f"User: {item['instruction']}\n{item['input']}\nAssistant: {item['output']}{tokenizer.eos_token}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
        
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        
        g_vecs = []
        for name, param in model.named_parameters():
             if "lora" in name and param.grad is not None:
                g_vecs.append(param.grad.view(-1).cpu().float())
        
        if g_vecs:
            grads.append(torch.cat(g_vecs))
        else:
            grads.append(torch.zeros(1))
        model.zero_grad()
    
    del model
    torch.cuda.empty_cache()
    
    if not grads: return torch.zeros((len(indices), 1))
    return torch.stack(grads)

def extract_representation_features(model_path, dataset_list, indices):
    """
    [New] RepSim 方法：提取特征表示 (Last Token Hidden State)
    """
    print(f"🧠 提取特征表示 (RepSim)... 样本数: {len(indices)}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # 注意：提取特征不需要 Peft/LoRA，只需要 Base Model
    # 因为我们看的是“预训练模型认为这两句话像不像”
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    model.eval()
    
    reps = []
    subset = [dataset_list[i] for i in indices]
    MAX_LEN = 256
    
    with torch.no_grad():
        for item in tqdm(subset, desc="Reps"):
            text = f"User: {item['instruction']}\n{item['input']}\nAssistant: {item['output']}{tokenizer.eos_token}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
            
            # output_hidden_states=True
            outputs = model(**inputs, output_hidden_states=True)
            
            # 取最后一层 (Last Hidden State)
            # shape: (1, seq_len, hidden_dim)
            last_hidden_state = outputs.hidden_states[-1]
            
            # 取最后一个有效 Token 的向量 (EOS 这里)
            # inputs['attention_mask'] 是 (1, seq_len)
            # 找到最后一个 1 的位置
            seq_len = inputs['attention_mask'].sum(dim=1) - 1
            # shape: (1, hidden_dim) -> (hidden_dim)
            last_token_rep = last_hidden_state[0, seq_len[0], :].float().cpu() # 转 float32
            
            reps.append(last_token_rep)
    
    del model
    torch.cuda.empty_cache()
    
    if not reps: return torch.zeros((len(indices), 1))
    return torch.stack(reps)

def calculate_shapley(model_path, dataset_list, oracle_data):
    n_oracle = min(len(oracle_data), CONFIG['n_val_samples'])
    oracle_subset = oracle_data[:n_oracle]
    
    print(f"🔧 Shapley计算 (Gradient-based): Train={len(dataset_list)}, Val={len(oracle_subset)}")
    
    train_grads = extract_gradient_features(model_path, dataset_list, list(range(len(dataset_list))))
    val_grads = extract_gradient_features(model_path, oracle_subset, list(range(len(oracle_subset))))
    
    min_len = min(train_grads.shape[1], val_grads.shape[1])
    return compute_knn_shapley_gradient(train_grads[:, :min_len], val_grads[:, :min_len], K=5)

def calculate_repsim_scores(model_path, dataset_list, oracle_data):
    """
    计算 RepSim 的两种分数 (一次提取，两种计算，高效对比)：
    1. Mean Cosine Similarity (RepSim-Mean)
    2. KNN-Shapley (RepSim-KNN)
    """
    n_oracle = min(len(oracle_data), CONFIG['n_val_samples'])
    oracle_subset = oracle_data[:n_oracle]
    
    print(f"🧩 提取特征 (RepSim)... Train={len(dataset_list)}, Val={len(oracle_subset)}")
    
    # 1. 提取特征 (Input X Vector，只提取一次)
    train_reps = extract_representation_features(model_path, dataset_list, list(range(len(dataset_list))))
    val_reps = extract_representation_features(model_path, oracle_subset, list(range(len(oracle_subset))))
    
    # ensure float32
    train_reps = train_reps.float()
    val_reps = val_reps.float()

    # 2. 计算 RepSim-Mean (Feature-based Mean Cosine)
    print(f"   -> [1/2] 计算 RepSim-Mean (Cosine)...")
    train_norm = F.normalize(train_reps, p=2, dim=1)
    val_norm = F.normalize(val_reps, p=2, dim=1)
    sim_matrix = torch.matmul(train_norm, val_norm.T)
    scores_mean = sim_matrix.mean(dim=1).numpy()

    # 3. 计算 RepSim-KNN (Feature-based KNN Shapley)
    print(f"   -> [2/2] 计算 RepSim-KNN (Shapley)...")
    min_dim = min(train_reps.shape[1], val_reps.shape[1])
    #复用通用的 KNN 算子，输入 Feature
    scores_knn = compute_knn_shapley_gradient(train_reps[:, :min_dim], val_reps[:, :min_dim], K=5)
    
    return scores_mean, scores_knn

# ==========================================
# 3. 训练与评估
# ==========================================

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = f"User: {item['instruction']}\n{item['input']}\nAssistant: {item['output']}{self.tokenizer.eos_token}"
        
        # 强制 Pad 到最大长度，避免 DataCollator 报错
        tokenized = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_len, 
            return_tensors="pt"
        )
        
        # squeeze() 去掉 batch 维度 (1, L) -> (L)
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # 对于 Causal LM，labels 就是 input_ids
        # 将 padding 部分的 label 设为 -100 (忽略计算 loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def run_sft_training(model_path, dataset_list, run_name):
    # 如果数据集过小，跳过
    if len(dataset_list) == 0:
        print(f"⚠️ {run_name} 数据集为空，跳过训练。")
        return

    output_path = os.path.join(CONFIG['output_dir'], run_name)
    print(f"\n🚀 [Training] 开始训练: {run_name}")
    print(f"   样本数量: {len(dataset_list)}")
    print(f"   保存路径: {output_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # 使用自定义 Dataset
    train_dataset = SFTDataset(dataset_list, tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"]))
    
    args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=8,   # [Fix OOM] 到8，配合累积梯度
        gradient_accumulation_steps=4,   # [Fix OOM] 8*4=32 等效 Batch Size
        num_train_epochs=3,      
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no", 
        report_to="none",
        fp16=True,                       
        dataloader_num_workers=2,        # [Fix] 减少 worker 防止多进程内存开销
    )
    
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_dataset, 
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    
    # 训练结束后保存一次
    trainer.save_model(output_path)
    
    # --- ROUGE-L 简易评估 ---
    print(f"📏 [Eval] {run_name} ROUGE Check...")
    try:
        model.eval()
        # 取前 10 个样本作为测试
        # 为了公平，我们应该用固定的、干净的测试集? 
        # 这里 Demo 简单起见，我们用 dataset_list 的前 10 个。
        test_samples = dataset_list[:10]
        preds, refs = [], []
        
        def calculate_local_rouge(pred_str, ref_str):
            x = list(pred_str.strip())
            y = list(ref_str.strip())
            if not x or not y: return 0.0
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            lcs_len = dp[m][n]
            if (len(x) + len(y)) == 0: return 0.0
            return 2.0 * lcs_len / (len(x) + len(y))

        for item in test_samples:
            prompt = f"User: {item['instruction']}\n\nAssistant: "
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant: ")[-1].strip()
            preds.append(pred)
            refs.append(item['output'])
        
        scores = [calculate_local_rouge(p, r) for p, r in zip(preds, refs)]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"📊 {run_name} Avg ROUGE-L: {avg_score:.4f}")
        
    except Exception as e:
        print(f"⚠️ Eval Error: {e}")
    
    # 清理显存
    del model, trainer
    torch.cuda.empty_cache()

# ==========================================
# 主流程
# ==========================================
def main(args=None):
    if args is None:
        args = parse_args()

    if args.n_samples is not None:
        CONFIG['n_samples'] = args.n_samples
    if args.poison_ratio is not None:
        CONFIG['poison_ratio'] = args.poison_ratio
    if args.n_val_samples is not None:
        CONFIG['n_val_samples'] = args.n_val_samples
    if args.seed is not None:
        CONFIG['seed'] = args.seed

    dataset_path, dataset_name = resolve_dataset_path(args.dataset, args.dataset_path)
    if args.output_subdir:
        CONFIG['output_dir'] = os.path.join(CURRENT_DIR, "server_results", args.output_subdir)
    elif dataset_name != "alpaca_local":
        CONFIG['output_dir'] = os.path.join(CURRENT_DIR, "server_results", dataset_name)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print(f"🌟 SFT Server Persistent Demo (Multi-Model Comparison) 启动")
    print(f"🗂️ 使用数据集: {dataset_name}")
    print(f"📁 数据文件: {dataset_path}")
    print(f"📦 输出目录: {CONFIG['output_dir']}")
    
    model_path = get_local_model_path()
    
    # 1. 准备数据
    raw_dirty, raw_pure, dirty_indices_gt, oracle_data = prepare_data_local(dataset_path)
    
    # ----------------------------------------------------
    # 2. 计算 Gradient Shapley (Ours)
    # ----------------------------------------------------
    sv = calculate_shapley(model_path, raw_dirty, oracle_data)
    
    n_remove = int(len(raw_dirty) * CONFIG['poison_ratio'])
    keep_indices = np.argsort(sv)[n_remove:]
    cleaned_data = [raw_dirty[i] for i in keep_indices]
    
    removed_indices = np.argsort(sv)[:n_remove]
    recall = len(set(removed_indices).intersection(set(dirty_indices_gt))) / (len(dirty_indices_gt) + 1e-9)
    print(f"✅ [Gradient] Shapley 清洗 Recall: {recall:.2%}")

    # ==========================
    # 3.5 计算 RepSim 并清洗 (新增对照组: Mean & KNN)
    # ==========================
    # 一次性计算两种特征基准分数
    repsim_scores_mean, repsim_scores_knn = calculate_repsim_scores(model_path, raw_dirty, oracle_data)
    
    # --- A. RepSim - Mean ---
    keep_indices_mean = np.argsort(repsim_scores_mean)[n_remove:]
    cleaned_data_repsim_mean = [raw_dirty[i] for i in keep_indices_mean]
    
    removed_indices_mean = np.argsort(repsim_scores_mean)[:n_remove]
    recall_mean = len(set(removed_indices_mean).intersection(set(dirty_indices_gt))) / (len(dirty_indices_gt) + 1e-9)
    print(f"✅ [RepSim-Mean] 清洗 Recall: {recall_mean:.2%}")

    # --- B. RepSim - KNN ---
    keep_indices_knn = np.argsort(repsim_scores_knn)[n_remove:]
    cleaned_data_repsim_knn = [raw_dirty[i] for i in keep_indices_knn]
    
    removed_indices_knn = np.argsort(repsim_scores_knn)[:n_remove]
    recall_knn = len(set(removed_indices_knn).intersection(set(dirty_indices_gt))) / (len(dirty_indices_gt) + 1e-9)
    print(f"✅ [RepSim-KNN] 清洗 Recall: {recall_knn:.2%}")

    # 4. 对比训练
    print("\n⚔️ 开始五组模型对比训练 ⚔️")
    print("------------------------------------------------")
    
    # A. 脏模型 (Dirty Model)
    run_sft_training(model_path, raw_dirty, "dirty_model")
    
    # B. 理想模型 (Oracle Model)
    run_sft_training(model_path, raw_pure, "oracle_model")
    
    # C. 我们的模型 (Clean Model - Gradient-KNN)
    run_sft_training(model_path, cleaned_data, "clean_model_gradient_knn")
    
    # D. 对照模型 1 (Clean Model - RepSim Mean)
    run_sft_training(model_path, cleaned_data_repsim_mean, "clean_model_repsim_mean")
    
    # E. 对照模型 2 (Clean Model - RepSim KNN)
    run_sft_training(model_path, cleaned_data_repsim_knn, "clean_model_repsim_knn")
    
    print("\n🎉 所有实验完成! 请查看上方的 ROUGE 分数差异。")

if __name__ == "__main__":
    main()
 