import torch

def display_gpu_info():
    gpu_stats = torch.cuda.get_device_properties(0)
    reserved_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{reserved_memory} GB of memory reserved.")
