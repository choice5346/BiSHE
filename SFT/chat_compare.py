import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =================配置=================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # d:\BiSHE
DIRTY_PATH = os.path.join(ROOT_DIR, "SFT", "results", "dirty_model")
CLEAN_PATH = os.path.join(ROOT_DIR, "SFT", "results", "clean_model")
ORACLE_PATH = os.path.join(ROOT_DIR, "SFT", "results", "oracle_model")
# ======================================

def load_model(path, name):
    print(f"⏳ 正在加载 {name} ... ({path})")
    try:
        # 强制使用 cpu 或者 cuda，这里我们为了规避偶发的 TensorCompare 错误，先尝试 safe load
        # 但通常这是因为 embedding 溢出或者 token id 问题。
        # 我们这里暂时保持 cuda，但加一个设置。
        tokenizer = AutoTokenizer.from_pretrained(path)
        # Qwen 的词表很大，有时需要 resize
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.float16)
        
        # ⚠️ 关键修复：Qwen1.5 有时候 eos_token_id 可能会出问题，显式设置 pad
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
            
        return tokenizer, model
    except Exception as e:
        print(f"❌ 加载 {name} 失败: {e}")
        return None, None

def generate_response(model, tokenizer, instruction):
    # 构造与训练时一致的 Prompt
    # 训练格式: User: {instruction}\n{input}\nAssistant: {output}
    prompt = f"User: {instruction}\n\nAssistant: "
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 手动处理 attention_mask (安全起见)
    # inputs 包含 input_ids 和 attention_mask
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True,      # 允许采样
            temperature=0.7,     # 温度
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1 # 避免复读机
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 Assistant 之后的部分
    if "Assistant: " in response:
        response = response.split("Assistant: ")[1].strip()
    return response

def main():
    print("="*50)
    print("🤖 SFT 模型对比对话系统 (Dirty vs Clean)")
    print("="*50)

    # 1. 检查模型文件是否存在
    if not os.path.exists(DIRTY_PATH) or not os.path.exists(CLEAN_PATH):
        print("⚠️ 警告：未找到模型文件！")
        print(f"请检查路径:\n  {DIRTY_PATH}\n  {CLEAN_PATH}")
        print("💡 提示：您之前的 sft_demo.py 可能没有保存模型，请重新运行一次 sft_demo.py")
        return

    # 2. 加载模型
    # 考虑到显存，我们假设 6GB 能同时放下两个 0.5B 模型 (约 2-3GB)
    # 如果爆显存，可以改成加载一个 -> 对话 -> 卸载 -> 加载另一个，但那样太慢
    tk_dirty, model_dirty = load_model(DIRTY_PATH, "Dirty Model (脏数据训练)")
    if not model_dirty: return
    
    tk_clean, model_clean = load_model(CLEAN_PATH, "Clean Model (清洗后训练)")
    if not model_clean: return
    
    tk_oracle, model_oracle = load_model(ORACLE_PATH, "Oracle Model (原始纯净数据)")
    # Oracle 可选，如果没有就不加载
    if not model_oracle: 
        print("⚠️ 提示：未找到 Oracle 模型，将只对比 Dirty vs Clean")

    print("\n✅ 模型加载完成！请输入问题进行测试 (输入 'q' 退出)")
    print("-" * 50)

    while True:
        query = input("\n🗣️  User: ")
        if query.lower() in ['q', 'quit', 'exit']:
            break
            
        if not query.strip():
            continue
            
        print("\n" + "-"*20 + " 生成中 " + "-"*20)
        
        # 生成 Dirty
        ans_dirty = generate_response(model_dirty, tk_dirty, query)
        print(f"\n💩 Dirty Model (基线):\n{ans_dirty}")
        
        # 生成 Clean
        ans_clean = generate_response(model_clean, tk_clean, query)
        print(f"\n✨ Clean Model (你的算法):\n{ans_clean}")
        
        # 生成 Oracle
        if model_oracle:
            ans_oracle = generate_response(model_oracle, tk_oracle, query)
            print(f"\n🌟 Oracle Model (天花板):\n{ans_oracle}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
