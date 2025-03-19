import torch 


try:
    torch_version = torch.__version__
    print(f"PyTorch is available. Version: {torch_version}")
except ImportError:
    raise ImportError("PyTorch is not available. Please install it to proceed.")



if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install it to proceed.")


# Check and list available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No GPUs are available. Please ensure you have a GPU to proceed.")
else:
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")



print("\nCkeck complete, basic system / environment requirements fullfilled.")