import torch
import os

# Check if CUDA is available
print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Check how many GPUs are detected
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs in KandaEnv!")
    
    for i in range(gpu_count):
        # Print the name of each GPU
        print(f" -> GPU {i}: {torch.cuda.get_device_name(i)}")

    print("-" * 30)
    
    # [Critical Test] Try to use GPU 1 (Since GPU 0 is busy)
    try:
        target_device = "cuda:1" 
        print(f"Attempting to send data to {target_device} (Tesla V100)...")
        
        # Create a random matrix
        x = torch.rand(1000, 1000)
        # Move it to the GPU
        x_gpu = x.to(target_device)
        
        # Perform matrix multiplication on GPU (This is what GPUs are good at)
        result = torch.matmul(x_gpu, x_gpu)
        
        print(f"[SUCCESS] Matrix multiplication completed on GPU 1.")
        print(f"Result device: {result.device}")
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

else:
    print("[BAD NEWS] PyTorch did not find any GPU, running on CPU maybe?")