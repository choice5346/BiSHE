import os
import sys
import json
import random
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

# 定义本地资源保存路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 资源保存在当前脚本同级目录下的 server_resources
RESOURCES_DIR = os.path.join(CURRENT_DIR, "server_resources")
# 这里直接读取 alpaca_data.json 文件
DATASET_PATH = os.path.join(RESOURCES_DIR, "alpaca_data.json")
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

def prepare_data_local():
    """
    加载数据并进行切分、投毒
    返回:
    1. final_data (投毒后的训练集 -> 对应 'Dirty Model')
    2. pure_data (未投毒的纯净训练集 -> 对应 'Oracle Model')
    3. dirty_indices_gt (投毒索引)
    4. oracle_data (用于 Shapley 计算的验证集)
    """
    print("📥 正在读取数据文件...")
    
    ds_full = None
    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, 'r', encoding='utf-8') as f:
                ds_full = json.load(f)
            print(f"✅ 成功加载 JSON 数据: {len(ds_full)} 条")
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

def compute_knn_shapley_gradient(train_grads, val_grads, K=10):
    """
    计算 KNN-Shapley 值
    """
    N_train = train_grads.shape[0]
    N_val = val_grads.shape[0]
    
    # 归一化
    train_grads = F.normalize(train_grads, p=2, dim=1)
    val_grads = F.normalize(val_grads, p=2, dim=1)
    
    print(f"   -> 计算相似度矩阵 (CPU)...")
    val_cpu = val_grads.cpu()
    train_cpu = train_grads.cpu()
    S = torch.matmul(val_cpu, train_cpu.T).numpy()
    
    shapley_values = np.zeros(N_train)
    
    for j in range(N_val):
        s_row = S[j]
        topk_indices = np.argsort(s_row)[-K:]
        shapley_values[topk_indices] += s_row[topk_indices]
        
    shapley_values /= N_val
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

def calculate_shapley(model_path, dataset_list, oracle_data):
    n_oracle = min(len(oracle_data), CONFIG['n_val_samples'])
    oracle_subset = oracle_data[:n_oracle]
    
    print(f"🔧 Shapley计算: Train={len(dataset_list)}, Val={len(oracle_subset)}")
    
    train_grads = extract_gradient_features(model_path, dataset_list, list(range(len(dataset_list))))
    val_grads = extract_gradient_features(model_path, oracle_subset, list(range(len(oracle_subset))))
    
    min_len = min(train_grads.shape[1], val_grads.shape[1])
    return compute_knn_shapley_gradient(train_grads[:, :min_len], val_grads[:, :min_len], K=5)

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
        tokenized = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors=None)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy()
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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,      
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no", # 本地Demo节省空间不保存每轮checkpoint
        report_to="none",
        fp16=False,
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
def main():
    print(f"🌟 SFT Server Persistent Demo (Multi-Model Comparison) 启动")
    
    model_path = get_local_model_path()
    
    # 1. 准备数据
    # raw_dirty: 包含投毒的数据 (对应 Baseline)
    # raw_pure:  原本的干净数据 (对应 Oracle)
    raw_dirty, raw_pure, dirty_indices_gt, oracle_data = prepare_data_local()
    
    # 2. 计算 Shapley 并清洗
    # 计算是基于 raw_dirty 进行筛选
    sv = calculate_shapley(model_path, raw_dirty, oracle_data)
    
    n_remove = int(len(raw_dirty) * CONFIG['poison_ratio'])
    keep_indices = np.argsort(sv)[n_remove:]
    # cleaned_data: 算法清洗后的数据 (对应 Clean/Ours)
    cleaned_data = [raw_dirty[i] for i in keep_indices]
    
    # 3. 计算 Recall
    removed_indices = np.argsort(sv)[:n_remove]
    recall = len(set(removed_indices).intersection(set(dirty_indices_gt))) / (len(dirty_indices_gt) + 1e-9)
    print(f"✅ Shapley 清洗 Recall: {recall:.2%}")

    # 4. 对比训练
    print("\n⚔️ 开始三组模型对比训练 ⚔️")
    print("------------------------------------------------")
    
    # A. 脏模型 (Dirty Model) - 用被投毒的数据练
    run_sft_training(model_path, raw_dirty, "dirty_model")
    
    # B. 理想模型 (Oracle Model) - 用未拆封的干净数据练 (上限)
    run_sft_training(model_path, raw_pure, "oracle_model")
    
    # C. 我们的模型 (Clean Model) - 用 Shapley 洗过的数据练
    run_sft_training(model_path, cleaned_data, "clean_model")
    
    print("\n🎉 所有实验完成! 请查看上方的 ROUGE 分数差异。")

if __name__ == "__main__":
    main()
