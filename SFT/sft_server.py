import os
import sys
import json
import random
import numpy as np
import torch
import shutil
from datasets import load_dataset, Dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import evaluate
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
DATASET_PATH = os.path.join(RESOURCES_DIR, "alpaca_data")
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
# 1. 资源下载与准备 (核心修改部分)
# ==========================================

def get_local_model_path():
    """
    检查本地是否有模型，没有则下载 (优先尝试 ModelScope，其次 HuggingFace)
    """
    # 1. 检查指定目录下是否有 config.json，如果有说明已经下载过了
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print(f"✅ 检测到本地模型已存在: {MODEL_PATH}")
        return MODEL_PATH
    
    print(f"📥 本地未找到模型，开始下载...")
    print(f"   目标路径: {MODEL_PATH}")

    # 2. 尝试使用 ModelScope 下载 (国内最快)
    try:
        print("🚀 尝试使用 ModelScope 下载 (国内推荐)...")
        from modelscope import snapshot_download
        # ModelScope 下载后会返回具体的缓存路径
        mw_path = snapshot_download(CONFIG['model_id_ms'])
        
        # 将下载的文件复制/移动到我们指定的 MODEL_PATH
        print(f"   ModelScope 下载完成，正在同步到 {MODEL_PATH} ...")
        
        # 如果目标文件夹存在且非空，先清空
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
        
        # 复制
        shutil.copytree(mw_path, MODEL_PATH)
        print(f"✅ 模型已就绪: {MODEL_PATH}")
        return MODEL_PATH
    except ImportError:
        print("⚠️ 未安装 modelscope 库，跳过 ModelScope 下载方式。(建议 pip install modelscope)")
    except Exception as e:
        print(f"❌ ModelScope 下载失败: {e}")

    # 3. 尝试使用 HuggingFace 下载 (使用镜像)
    try:
        print("☁️ 尝试使用 HuggingFace (hf-mirror) 下载...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=CONFIG['model_id_hf'],
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,  
            resume_download=True
        )
        return MODEL_PATH
    except Exception as e:
        print(f"❌ HuggingFace 下载失败: {e}")
        raise RuntimeError("无法下载模型，请检查网络或手动下载模型到 server_resources/qwen_model 目录")

def prepare_data_local():
    """
    数据本地化加载逻辑
    """
    print("📥 正在加载数据...")
    ds_full = None
    
    # 1. 优先加载本地
    if os.path.exists(DATASET_PATH):
        try:
            print(f"📂 加载本地数据集: {DATASET_PATH}")
            from datasets import load_from_disk
            ds_loaded = load_from_disk(DATASET_PATH)
            if isinstance(ds_loaded, dict) or hasattr(ds_loaded, 'keys'):
                ds_full = ds_loaded['train'] if 'train' in ds_loaded else list(ds_loaded.values())[0]
            else:
                ds_full = ds_loaded
            print(f"✅ 本地数据加载成功! 总量: {len(ds_full)}")
        except Exception as e:
            print(f"❌ 本地数据损坏: {e}")
            
    # 2. 如果本地没有，尝试下载 (优先 ModelScope/HF)
    if ds_full is None:
        print("☁️ 正在下载 tatsu-lab/alpaca 数据集...")
        
        # --- 方案 A: 使用 ModelScope 下载 (国内最快) ---
        try:
            print("   [Attempt 1] 尝试 ModelScope (AI-ModelScope/alpaca-gpt4-data-en)...")
            from modelscope.msdatasets import MsDataset
            # ModelScope 上的 Alpaca 数据集 (英文版)
            ms_ds = MsDataset.load('AI-ModelScope/alpaca-gpt4-data-en', split='train')
            # 转换为 HuggingFace 格式 List[Dict]
            ds_full = []
            print("   -> 正在转换数据格式...")
            for item in ms_ds:
                ds_full.append({
                    'instruction': item.get('instruction', ''),
                    'input': item.get('input', ''),
                    'output': item.get('output', '')
                })
            print(f"✅ ModelScope 下载并转换成功! 条数: {len(ds_full)}")
        except Exception as e:
            print(f"⚠️ ModelScope 下载失败: {e}")

        # --- 方案 B: 使用 HF 镜像下载 (备选) ---
        if ds_full is None:
            try:
                print("   [Attempt 2] 尝试 HuggingFace 镜像 (hf-mirror.com)...")
                # 设置环境变量强制走此镜像
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                ds_full = load_dataset("tatsu-lab/alpaca", split="train")
                print("✅ HF 镜像下载成功!")
            except Exception as e:
                print(f"⚠️ HF 镜像下载失败: {e}")
        
        # --- 保存到本地 ---
        if ds_full is not None:
            try:
                print(f"💾 正在保存数据集到本地: {DATASET_PATH} ...")
                # 如果是 List，先转 Dataset
                if isinstance(ds_full, list):
                    Dataset.from_list(ds_full).save_to_disk(DATASET_PATH)
                else:
                    ds_full.save_to_disk(DATASET_PATH)
                print("✅ 数据集已持久化保存。")
            except Exception as e:
                print(f"⚠️ 保存失败: {e}")
        
        # --- 最终兜底 ---
        if ds_full is None:
            print("☢️ 所有下载方式均失败，使用合成数据兜底...")
            ds_full = [{"instruction": f"Solve {k}+{k}", "input":"", "output":f"{k+k}"} for k in range(5000)]

    # 3. 切分数据
    current_n = CONFIG['n_samples']
    print(f"✂️ 正在截取前 {current_n} 条数据用于本次实验...")
    
    ds_list = []
    count = 0
    for item in ds_full:
        ds_list.append({"instruction": item["instruction"], "input": item["input"], "output": item["output"]})
        count += 1
        if count >= current_n: break
            
    # 4. 投毒
    final_data = []
    dirty_indices_gt = [] 
    
    # Oracle 从原始数据里取 100 条 (不投毒)
    # 确保 oracle_data 不受投毒影响
    oracle_data = [x.copy() for x in ds_list[:100]] 
    
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
    
    print(f"✅ 最终训练数据: {len(final_data)} 条 | 验证数据(Oracle): {len(oracle_data)} 条")
    return final_data, dirty_indices_gt, oracle_data

# ==========================================
# 2. 梯度与 KNN-Shapley (优化版)
# ==========================================

def compute_knn_shapley_gradient(train_grads, val_grads, K=10):
    N_train = train_grads.shape[0]
    N_val = val_grads.shape[0]
    
    print(f"   -> 归一化...")
    train_grads = F.normalize(train_grads, p=2, dim=1)
    val_grads = F.normalize(val_grads, p=2, dim=1)
    
    print(f"   -> 计算相似度矩阵 (CPU)...")
    # 移至 CPU 计算避免 OOM
    val_cpu = val_grads.cpu()
    train_cpu = train_grads.cpu()
    S = torch.matmul(val_cpu, train_cpu.T).numpy()
    
    shapley_values = np.zeros(N_train)
    
    print(f"   -> KNN 估值...")
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
    # 确保 oracle_data 不会太多把内存撑爆
    n_oracle = min(len(oracle_data), CONFIG['n_val_samples'])
    oracle_subset = oracle_data[:n_oracle]
    
    print(f"🔧 准备计算: Train={len(dataset_list)}, Val={len(oracle_subset)}")
    
    train_grads = extract_gradient_features(model_path, dataset_list, list(range(len(dataset_list))))
    val_grads = extract_gradient_features(model_path, oracle_subset, list(range(len(oracle_subset))))
    
    min_len = min(train_grads.shape[1], val_grads.shape[1])
    return compute_knn_shapley_gradient(train_grads[:, :min_len], val_grads[:, :min_len], K=5)

# ==========================================
# 3. 训练与评估
# ==========================================
def run_sft_training(model_path, dataset_list, run_name):
    output_path = os.path.join(CONFIG['output_dir'], run_name)
    print(f"\n🚀 开始训练: {run_name} -> {output_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    hf_dataset = Dataset.from_list(dataset_list)
    hf_dataset = hf_dataset.map(lambda x: tokenizer(f"User: {x['instruction']}\n{x['input']}\nAssistant: {x['output']}{tokenizer.eos_token}", truncation=True, max_length=256), batched=False)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"]))
    
    args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        fp16=False,
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=hf_dataset, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(output_path)
    print(f"💾 模型已保存到: {output_path}")
    
    # 简单的 ROUGE 评估 (本地离线版 - 无需联网)
    print("📏 ROUGE Check (Offline)...")
    try:
        model.eval()
        test_samples = dataset_list[:10]
        preds, refs = [], []
        
        # --- 本地简易计算 ROUGE-L (基于字符级 LCS) ---
        def calculate_local_rouge(pred_str, ref_str):
            # 将字符串转为字符列表 (兼容中文和英文)
            x = list(pred_str.strip())
            y = list(ref_str.strip())
            if not x or not y: return 0.0
            
            # 动态规划计算最长公共子序列 (LCS)
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            lcs_len = dp[m][n]
            
            # 计算 F1 Score (ROUGE-L F1)
            # F1 = 2 * LCS / (len(pred) + len(ref))
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
        
        # 计算平均分
        scores = [calculate_local_rouge(p, r) for p, r in zip(preds, refs)]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"📊 {run_name} Manual-ROUGE-L: {avg_score:.4f}")
        
    except Exception as e:
        print(f"⚠️ Eval Error: {e}")

# ==========================================
# 主流程
# ==========================================
def main():
    print(f"🌟 SFT Server Persistent Demo (Updated) 启动")
    
    # 1. 准备本地模型 (由 ModelScope 驱动)
    model_path = get_local_model_path()
    
    # 2. 准备本地数据
    raw_data, dirty_indices_gt, oracle_data = prepare_data_local()
    
    # 3. 计算 & 清洗
    sv = calculate_shapley(model_path, raw_data, oracle_data)
    
    n_remove = int(len(raw_data) * CONFIG['poison_ratio'])
    keep_indices = np.argsort(sv)[n_remove:]
    cleaned_data = [raw_data[i] for i in keep_indices]
    
    # Check
    removed_indices = np.argsort(sv)[:n_remove]
    recall = len(set(removed_indices).intersection(set(dirty_indices_gt))) / (len(dirty_indices_gt) + 1e-9)
    print(f"✅ Recall: {recall:.2%}")

    # 4. 训练
    run_sft_training(model_path, cleaned_data, "clean_model")
    
    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
