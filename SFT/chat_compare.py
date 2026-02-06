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
    print("🤖 SFT 模型全量对比系统 (Dirty vs Clean vs Oracle)")
    print("="*50)

    # 1. 检查模型文件是否存在
    models_to_load = [
        ("Dirty Model (基线)", DIRTY_PATH, "dirty"),
        ("Clean Model (你的算法)", CLEAN_PATH, "clean"),
        ("Oracle Model (天花板)", ORACLE_PATH, "oracle")
    ]
    
    loaded_models = {}

    # 2. 加载模型
    for name, path, key in models_to_load:
        if os.path.exists(path):
            tokenizer, model = load_model(path, name)
            if model:
                loaded_models[key] = (tokenizer, model)
        else:
            print(f"⚠️ 跳过 {name}: 路径不存在")

    if not loaded_models:
        print("❌ 没有加载到任何模型，请先运行 sft_demo.py 训练模型")
        return

    print("\n✅ 模型加载完成！请输入问题进行测试 (输入 'q' 退出)")
    print("-" * 50)

    while True:
        query = input("\n🗣️  User: ")
        if query.lower() in ['q', 'quit', 'exit']:
            break
            
        if not query.strip():
            continue
            
        print("\n" + "-"*20 + " 生成中 " + "-"*20)
        
        # 依次生成回答
        for name, path, key in models_to_load:
            if key in loaded_models:
                tokenizer, model = loaded_models[key]
                try:
                    ans = generate_response(model, tokenizer, query)
                    # 使用不同的 emoji 区分
                    icon = "💩" if key == "dirty" else ("✨" if key == "clean" else "🌟")
                    print(f"\n{icon} {name}:\n{ans}")
                except Exception as e:
                    print(f"\n❌ {name} 生成出错: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
