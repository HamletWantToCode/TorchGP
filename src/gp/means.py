import torch
import torch.nn as nn

__all__ = ["ZeroScalarMean", "ZeroVectorMean"]

class BaseMean(nn.Module):
    def __init__(self):
        super().__init__()

class ZeroScalarMean(BaseMean):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        X: n_samples * n_features
        """
        device = X.device
        n_samples = X.shape[0]
        return torch.zeros(n_samples, device=device)

class ZeroVectorMean(BaseMean):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        """
        X: n_samples * n_features
        """
        device = X.device
        n_samples, n_features = X.shape
        return torch.zeros(n_samples*n_features, device=device)
