from datetime import datetime
import os

import torch


def print_torch_cuda_diagnostics(context=""):
    """Print CUDA discovery state without assuming that a device is usable."""

    prefix = f" ({context})" if context else ""
    details = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }

    for name, query in (
        ("available", torch.cuda.is_available),
        ("initialized", torch.cuda.is_initialized),
        ("device_count", torch.cuda.device_count),
        ("current_device", torch.cuda.current_device),
    ):
        try:
            details[name] = query()
        except Exception as exc:  # Diagnostics must never hide the original problem.
            details[name] = f"ERROR {type(exc).__name__}: {exc}"

    rendered = ", ".join(f"{key}={value}" for key, value in details.items())
    print(f"    CUDA diagnostics{prefix}: {rendered}", flush=True)
    return details

def handle_torch_device(device):
    """
    Utility function to handle torch device selection and print info about the device being used.
        Parameters:
            device (str or torch.device): Desired device, e.g. 'cuda', 'cpu', or specific GPU like 'cuda:0'.
        Returns:
            torch.device: The device that will be used for computations.
    """
    
    requested_device = device
    print_torch_cuda_diagnostics(f"handle_torch_device requested={requested_device}")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("    CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")

    if device.type == "cuda":
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("    CUDA requested but no CUDA devices are visible; falling back to CPU")
            return torch.device("cpu")

        if device.index is None:
            # A long-running process can retain a CUDA current-device index that is
            # no longer present (for example after GPU visibility changes).  Bare
            # "cuda" should still resolve to a valid visible device.
            try:
                idx = torch.cuda.current_device()
            except (AssertionError, RuntimeError):
                idx = 0
            if not 0 <= idx < device_count:
                idx = 0
            device = torch.device("cuda", idx)
        elif not 0 <= device.index < device_count:
            raise ValueError(
                f"CUDA device index {device.index} is invalid; "
                f"{device_count} CUDA device(s) are visible"
            )

        idx = device.index
        print(f"    Torch using: {device}, GPU index: {idx}, GPU name: {torch.cuda.get_device_name(device)}")
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

    cuda_tensors = [
        tensor for tensor in tensors.values()
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda
    ]
    if cuda_tensors and torch.cuda.device_count() > 0:
        memory_device = cuda_tensors[0].device
        try:
            total_gpu = torch.cuda.get_device_properties(memory_device).total_memory / 1024**2
            reserved = torch.cuda.memory_reserved(memory_device) / 1024**2
        except (AssertionError, RuntimeError) as exc:
            print(f"| CUDA memory summary unavailable: {type(exc).__name__}: {exc}")
            print_torch_cuda_diagnostics("print_cuda_tensors_mem failure")
            return

        tensor_percent = (total_tensor_mb / total_gpu) * 100 if total_gpu > 0 else 0
        reserved_percent = (reserved / total_gpu) * 100 if total_gpu > 0 else 0

        print(
            f"| TENSORS: {total_tensor_mb:8.2f} MB ({tensor_percent:5.1f}%) | "
            f"RESERVED: {reserved:8.2f} MB ({reserved_percent:5.1f}%) | "
            f"TOTAL GPU: {total_gpu:8.2f} MB  "
        )
    else:
        print(f"| CUDA memory summary skipped (no CUDA tensors or visible devices)")
        print(f"| TENSORS: {total_tensor_mb:8.2f} MB\n")
