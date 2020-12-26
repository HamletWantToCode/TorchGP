import torch 
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import grad

def pairwise(X1: torch.tensor, X2: torch.tensor):
    """
    X1: n_samples * ... * n_feature
    X2:
    """
    _X1 = X1[:, None]
    _X2 = X2[None, :]
    pw_diff = _X1 - _X2
    pw_distance = torch.sqrt(torch.sum(torch.pow(pw_diff, 2), dim=-1))
    return pw_distance

class BaseKernel(nn.Module):
    def __init__(self, overall_scaling: torch.tensor, character_length: torch.tensor):
        """
        overall_scaling: 1
        character_length: n_features or 1
        """
        super().__init__()
        self.c = Parameter(overall_scaling)
        self.l = Parameter(character_length) 
    

class Matern52(BaseKernel):
    def __init__(self, overall_scaling: torch.tensor, character_length: torch.tensor):
        super().__init__(overall_scaling, character_length)

    def forward(self, X1, X2):
        X1_l = X1 / self.l
        X2_l = X2 / self.l
        r = pairwise(X1_l, X2_l)
        
        sqrt_5 = math.sqrt(5)
        return torch.pow(self.c, 2) * (1.0 + sqrt_5*r + (5.0/3.0)*torch.pow(r, 2)) * torch.exp(-sqrt_5*r)

class Deriv1Matern52(BaseKernel):
    def __init__(self, overall_scaling: torch.tensor, character_length: torch.tensor):
        super().__init__(overall_scaling, character_length)
    
    def forward(self, X1, X2):
        X1_l = X1 / self.l
        X2_l = X2 / self.l
        DX1X2 = X1[:, None] - X2[None, :]
        r = pairwise(X1_l, X2_l)

        sqrt_5 = math.sqrt(5)
        fr = torch.pow(self.c, 2) * (-5.0/3.0)*torch.exp(-sqrt_5*r)*(1.0+sqrt_5*r)
        dx1x2_div_l2 = DX1X2 / torch.pow(self.l, 2)
        return torch.einsum("ij,ija->iaj", fr, dx1x2_div_l2)
        # TODO: reshape the 3-tensor into a matrix

class Deriv2Matern52(BaseKernel):
    def __init__(self, overall_scaling: torch.tensor, character_length: torch.tensor):
        super().__init__(overall_scaling, character_length)

    def forward(self, X1, X2):
        DX1X2 = X1[:, None] - X2[None, :]  # n_sample * n_sample * n_feature
        X1_l = X1 / self.l
        X2_l = X2 / self.l
        r = pairwise(X1_l, X2_l)     # n_sample * n_sample
        sqrt_5 = math.sqrt(5)

        fr = (5.0/3.0)*torch.exp(-sqrt_5*r)  # n_sample * n_sample
        part1 = torch.einsum("ij,ab->iajb", (1+sqrt_5*r), (1.0/torch.pow(self.l, 2))*torch.eye(len(self.l)))
        distance_div_l2 = torch.einsum("ija,ijb->iajb", DX1X2 / torch.pow(self.l, 2), DX1X2 / torch.pow(self.l, 2))
        k = torch.einsum("ij,iajb->iajb", fr, part1 - 5.0*distance_div_l2)
        return torch.pow(self.c, 2) * k
        # TODO: reshape the 4-tensor into a matrix






