import os
import math
import argparse
import json
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from sft_server import (
    get_local_model_path,
    resolve_dataset_path,
    CONFIG,
)

# Set standard paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(CURRENT_DIR, "server_resources", "qwen_model")
RESULTS_DIR = os.path.join(CURRENT_DIR, "server_results")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT adapters with FAIR, UNSEEN testing protocol")
    parser.add_argument("--dataset", type=str, default="alpaca_local", help="Dataset key in registry")
    parser.add_argument("--dataset_path", type=str, default=None, help="Custom dataset path")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples used in training")
    parser.add_argument("--test_samples", type=int, default=20, help="Number of unseen testing samples")
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Override result sub-directory under server_results",
    )
    return parser.parse_args()

def calculate_rouge_l(pred, ref):
    x = list(pred.strip())
    y = list(ref.strip())
    if not x or not y:
        return 0.0
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    if (len(x) + len(y)) == 0:
        return 0.0
    return 2.0 * lcs_len / (len(x) + len(y))

def normalize_text(s):
    return " ".join(s.strip().lower().split())

def exact_match(pred, ref):
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0

def char_f1(pred, ref):
    p = list(pred.strip())
    r = list(ref.strip())
    if not p or not r:
        return 0.0
    c_pred = Counter(p)
    c_ref = Counter(r)
    overlap = sum((c_pred & c_ref).values())
    precision = overlap / max(len(p), 1)
    recall = overlap / max(len(r), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def bleu1_char(pred, ref):
    p = list(pred.strip())
    r = list(ref.strip())
    if not p or not r:
        return 0.0
    c_pred = Counter(p)
    c_ref = Counter(r)
    overlap = sum((c_pred & c_ref).values())
    precision = overlap / max(len(p), 1)
    if len(p) > len(r):
        bp = 1.0
    else:
        bp = math.exp(1 - (len(r) / max(len(p), 1)))
    return bp * precision


def relative_length_error(pred, ref):
    lp = len(pred.strip())
    lr = len(ref.strip())
    if lr == 0:
        return 1.0 if lp > 0 else 0.0
    return abs(lp - lr) / lr


def evaluate_model(model_name, test_samples, tokenizer, result_base_dir):
    folder = os.path.join(result_base_dir, model_name)
    if not os.path.exists(folder):
        print(f"Skipping {model_name} (not found)")
        return None

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, folder)
    model.eval()

    metric_rouge, metric_em, metric_char_f1, metric_bleu1, metric_len_err = [], [], [], [], []

    for item in tqdm(test_samples, desc=f"Gen-{model_name}"):
        prompt = f"User: {item['instruction']}\n\nAssistant: "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant: ")[-1].strip()
        ref = item.get("output", "")

        metric_rouge.append(calculate_rouge_l(pred, ref))
        metric_em.append(exact_match(pred, ref))
        metric_char_f1.append(char_f1(pred, ref))
        metric_bleu1.append(bleu1_char(pred, ref))
        metric_len_err.append(relative_length_error(pred, ref))

    result = {
        "ROUGE_L": sum(metric_rouge) / len(metric_rouge),
        "EM": sum(metric_em) / len(metric_em),
        "CHAR_F1": sum(metric_char_f1) / len(metric_char_f1),
        "BLEU1_CHAR": sum(metric_bleu1) / len(metric_bleu1),
        "LEN_REL_ERR": sum(metric_len_err) / len(metric_len_err),
    }

    del model
    del base_model
    torch.cuda.empty_cache()
    return result

def evaluate(args=None):
    if args is None:
        args = parse_args()

    if args.n_samples is not None:
        CONFIG["n_samples"] = args.n_samples

    dataset_path, dataset_name = resolve_dataset_path(args.dataset, args.dataset_path)
    if args.output_subdir:
        result_base_dir = os.path.join(RESULTS_DIR, args.output_subdir)
    elif dataset_name != "alpaca_local":
        result_base_dir = os.path.join(RESULTS_DIR, dataset_name)
    else:
        result_base_dir = RESULTS_DIR

    print(f"Dataset key: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"Result dir: {result_base_dir}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("\n[Step 1] Loading pure, unseen test samples from JSON...")
    # 真正的绝对公平：从 JSON 中读取所有模型在训练期绝对没见过的全新未考数据
    with open(dataset_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    train_n = CONFIG["n_samples"]
    # 选取训练集范围之外的独立数据（例如训练用了0-1000条，那我们就拿1000条之后的数据作为全新测练题）
    unseen_test_samples = full_data[train_n : train_n + args.test_samples] 
    if len(unseen_test_samples) == 0:
        print("警告：数据集后面的数据不足，只能取末尾倒数来兜底，这可能仍然会导致数据泄漏。")
        unseen_test_samples = full_data[-args.test_samples:] # 数据量不够时的兜底逻辑

    all_results = {}
    ordered_models = [
        "dirty_model",
        "oracle_model",
        "clean_model_gradient_knn",
        "clean_model_repsim_mean",
        "clean_model_repsim_knn",
    ]

    print(f"\n[Step 2] Evaluate all models on a TRULY UNSEEN, FAIR test set ({len(unseen_test_samples)} samples)...")
    for model_name in ordered_models:
        test_samples = unseen_test_samples
        print(f"\nEvaluating {model_name} on {len(test_samples)} unseen samples...")
        try:
            metrics = evaluate_model(model_name, test_samples, tokenizer, result_base_dir)
            if metrics is None:
                continue
            all_results[model_name] = metrics
            print(
                "✅ "
                f"{model_name} | ROUGE-L={metrics['ROUGE_L']:.4f}, "
                f"EM={metrics['EM']:.4f}, "
                f"Char-F1={metrics['CHAR_F1']:.4f}, "
                f"BLEU1-Char={metrics['BLEU1_CHAR']:.4f}, "
                f"LenRelErr={metrics['LEN_REL_ERR']:.4f}"
            )
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    print(f"\n\nFINAL COMPARISON (FAIR, UNSEEN TEST SET) - Dataset: {dataset_name}:")
    print("-------------------------------------------------------------------")
    if not all_results:
        print("No available results.")
        return

    sorted_rouge = sorted(all_results.items(), key=lambda kv: kv[1]["ROUGE_L"], reverse=True)
    for name, m in sorted_rouge:
        print(
            f"{name:30s} | "
            f"ROUGE-L={m['ROUGE_L']:.4f} | "
            f"EM={m['EM']:.4f} | "
            f"Char-F1={m['CHAR_F1']:.4f} | "
            f"BLEU1-Char={m['BLEU1_CHAR']:.4f} | "
            f"LenRelErr={m['LEN_REL_ERR']:.4f}"
        )

if __name__ == "__main__":
    evaluate()