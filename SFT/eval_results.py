import os
import math
import argparse
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from sft_server import (
    get_local_model_path,
    prepare_data_local,
    calculate_shapley,
    calculate_repsim_scores,
    resolve_dataset_path,
    CONFIG,
)

# Set standard paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(CURRENT_DIR, "server_resources", "qwen_model")
RESULTS_DIR = os.path.join(CURRENT_DIR, "server_results")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT adapters with original protocol")
    parser.add_argument("--dataset", type=str, default="alpaca_local", help="Dataset key in registry")
    parser.add_argument("--dataset_path", type=str, default=None, help="Custom dataset path")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--poison_ratio", type=float, default=None, help="Poison ratio")
    parser.add_argument("--n_val_samples", type=int, default=None, help="Oracle validation size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
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


def build_original_logic_datasets(model_path, dataset_path):
    print("\n[Step 1] Rebuild datasets with original pipeline...")
    raw_dirty, raw_pure, _dirty_indices_gt, oracle_data = prepare_data_local(dataset_path)

    sv = calculate_shapley(model_path, raw_dirty, oracle_data)
    n_remove = int(len(raw_dirty) * CONFIG["poison_ratio"])

    keep_indices_grad = sv.argsort()[n_remove:]
    cleaned_gradient_knn = [raw_dirty[i] for i in keep_indices_grad]

    repsim_scores_mean, repsim_scores_knn = calculate_repsim_scores(model_path, raw_dirty, oracle_data)
    keep_indices_mean = repsim_scores_mean.argsort()[n_remove:]
    keep_indices_knn = repsim_scores_knn.argsort()[n_remove:]

    cleaned_repsim_mean = [raw_dirty[i] for i in keep_indices_mean]
    cleaned_repsim_knn = [raw_dirty[i] for i in keep_indices_knn]

    return {
        "dirty_model": raw_dirty,
        "oracle_model": raw_pure,
        "clean_model_gradient_knn": cleaned_gradient_knn,
        "clean_model_repsim_mean": cleaned_repsim_mean,
        "clean_model_repsim_knn": cleaned_repsim_knn,
    }


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
                max_new_tokens=50,
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
    if args.poison_ratio is not None:
        CONFIG["poison_ratio"] = args.poison_ratio
    if args.n_val_samples is not None:
        CONFIG["n_val_samples"] = args.n_val_samples
    if args.seed is not None:
        CONFIG["seed"] = args.seed

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

    model_path = get_local_model_path()
    datasets_map = build_original_logic_datasets(model_path, dataset_path)

    all_results = {}
    ordered_models = [
        "dirty_model",
        "oracle_model",
        "clean_model_gradient_knn",
        "clean_model_repsim_mean",
        "clean_model_repsim_knn",
    ]

    print("\n[Step 2] Evaluate each model on its own first-10 samples (original logic)...")
    for model_name in ordered_models:
        test_samples = datasets_map[model_name][:10]
        print(f"\nEvaluating {model_name} on its own 10 samples...")
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

    print("\n\nFINAL COMPARISON (Original Logic: each model on its own first-10):")
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
