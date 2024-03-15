import torch.backends.mps as mps
import torch.cuda as cuda

def get_device():
    if mps.is_available():
        return 'mps'
    return 'cuda' if cuda.is_available() else 'cpu'