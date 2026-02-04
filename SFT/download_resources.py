import os
# 强制使用 HF 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# 定义保存路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # d:\BiSHE
LOCAL_DIR = os.path.join(ROOT_DIR, "local_resources")

if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR)

print(f"📂 资源将保存到: {LOCAL_DIR}")

# 1. 下载 Alpaca 数据
print("\n[1/3] 正在下载 Alpaca 数据集 (约 25MB)...")
try:
    ds = load_dataset("tatsu-lab/alpaca")
    save_path = os.path.join(LOCAL_DIR, "alpaca_data")
    ds.save_to_disk(save_path)
    print(f"✅ 数据集已保存至: {save_path}")
except Exception as e:
    print(f"❌ 数据集下载失败: {e}")

# 2. 下载 Embedding 模型
print("\n[2/3] 正在下载 Embedding 模型 (all-MiniLM-L6-v2, 约 80MB)...")
try:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    save_path = os.path.join(LOCAL_DIR, "embed_model")
    model.save(save_path)
    print(f"✅ Embedding 模型已保存至: {save_path}")
except Exception as e:
    print(f"❌ Embedding 模型下载失败: {e}")

# 3. 下载 LLM 模型
print("\n[3/3] 正在下载 Qwen1.5-0.5B 模型 (约 1.2GB)...")
print("⏳ 这个过程可能需要几分钟，请耐心等待...")
try:
    model_id = "Qwen/Qwen1.5-0.5B"
    save_path = os.path.join(LOCAL_DIR, "qwen_model")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_path)
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.save_pretrained(save_path)
    print(f"✅ Qwen 模型已保存至: {save_path}")
except Exception as e:
    print(f"❌ Qwen 模型下载失败: {e}")

print("\n🎉 所有下载任务结束！")
