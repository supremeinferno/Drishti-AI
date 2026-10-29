import torch


def pick_device() -> str:
    """Use the laptop GPU if there is one (NVIDIA -> cuda, Apple Silicon -> mps)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
