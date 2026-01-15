import torch

def check_gpu():
    print("-" * 30)
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("CUDA is available! ✅")
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        print(f"Current CUDA device index: {torch.cuda.current_device()}")
    else:
        print("CUDA is NOT available. Using CPU. ❌")
        
    print("-" * 30)

if __name__ == "__main__":
    check_gpu()
