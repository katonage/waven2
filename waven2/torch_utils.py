import torch

def handle_torch_device(device):
    """
    Utility function to handle torch device selection and print info about the device being used.
        Parameters:
            device (str or torch.device): Desired device, e.g. 'cuda', 'cpu', or specific GPU like 'cuda:0'.
        Returns:
            torch.device: The device that will be used for computations.
    """
    
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("    CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")

    if device.type == "cuda":
        idx = torch.cuda.current_device()
        print(f"    Torch using: {device}, GPU index: {idx}, GPU name: {torch.cuda.get_device_name(idx)}")
        torch.cuda.empty_cache()
    else:
        print(f"    Torch using: {device}") 
    return device
    

import torch

def print_cuda_tensors_mem(tensors: dict):
    """
    Print GPU memory usage of given torch tensors, plus compact summary with % usage.
    
    Parameters
    ----------
    tensors : dict
        Dictionary {name: torch_tensor}
    """

    total_bytes = 0

    first_tensor = next(iter(tensors.values()), None)

    if isinstance(first_tensor, torch.Tensor):
        print(f"| Torch ({first_tensor.device}) tensor memory usage:")
    else:
        print("| Torch tensor memory usage:")
    
    print("| " + "-" * 60)

    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            continue
        if not t.is_cuda:
            continue

        bytes_ = t.numel() * t.element_size()
        total_bytes += bytes_

        print(f"| {name:20s}: {bytes_/1024**2:8.2f} MB  | shape={tuple(t.shape)}  dtype={t.dtype}")

    print("| " + "-" * 60)

    total_tensor_mb = total_bytes / 1024**2

    if torch.cuda.is_available():
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**2
        reserved  = torch.cuda.memory_reserved() / 1024**2

        tensor_percent = (total_tensor_mb / total_gpu) * 100 if total_gpu > 0 else 0
        reserved_percent = (reserved / total_gpu) * 100 if total_gpu > 0 else 0

        print(
            f"| TENSORS: {total_tensor_mb:8.2f} MB ({tensor_percent:5.1f}%) | "
            f"RESERVED: {reserved:8.2f} MB ({reserved_percent:5.1f}%) | "
            f"TOTAL GPU: {total_gpu:8.2f} MB  "
        )
    else:
        print(f"TENSORS: {total_tensor_mb:8.2f} MB\n")