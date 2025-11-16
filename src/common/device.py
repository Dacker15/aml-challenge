import torch


def get_device():
    """
    Get the available device
    If a GPU is available, return 'cuda'
    If a MPS device is available (Apple Silicon), return 'mps'
    Otherwise, return 'cpu'
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
