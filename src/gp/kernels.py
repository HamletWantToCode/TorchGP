import torch 
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from .utils import *

# TODO: Add SE kernel

class BaseKernel(nn.Module):
    def __init__(self, overall_scaling: torch.tensor, character_length: torch.tensor):
        """
        overall_scaling: 1
        character_length: n_features or 1
        """
        super().__init__()
        self.c = Parameter(overall_scaling)
        self.l = Parameter(character_length) 

    @staticmethod
    def _distance(X1: torch.tensor, X2: torch.tensor=None):
        if X2 is not None:
            return X1[:,None]-X2
        else:
            return X1[:,None]-X1
    
    @staticmethod
    def _r(X1: torch.tensor, X2: torch.tensor=None):
        if X2 is not None:
            return pairwise2(X1, X2)
        else:
            return pairwise1(X1)

    def _streched(self, X1: torch.tensor, X2: torch.tensor=None):
        if X2 is not None:
            return X1/self.l, X2/self.l
        else:
            return X1/self.l, None


class Matern52(BaseKernel):
    def __init__(self, overall_scaling: torch.tensor, character_length: torch.tensor):
        super().__init__(overall_scaling, character_length)

    def forward(self, X1, X2: torch.tensor=None):
        X1_l, X2_l = self._streched(X1, X2)
        r = self._r(X1_l, X2_l)
        sqrt_5 = math.sqrt(5)
        return torch.pow(self.c, 2) * (1.0 + sqrt_5*r + (5.0/3.0)*torch.pow(r, 2)) * torch.exp(-sqrt_5*r)


class Deriv1Matern52(BaseKernel):
    def __init__(self, overall_scaling: torch.tensor, character_length: torch.tensor):
        super().__init__(overall_scaling, character_length)
    
    def forward(self, X1, X2: torch.tensor=None):
        DX1X2 = self._distance(X1, X2)
        X1_l, X2_l = self._streched(X1, X2)
        r = self._r(X1_l, X2_l)
        sqrt_5 = math.sqrt(5)
        fr = torch.pow(self.c, 2) * (-5.0/3.0)*torch.exp(-sqrt_5*r)*(1.0+sqrt_5*r)
        dx1x2_div_l2 = DX1X2 / torch.pow(self.l, 2)
        return torch.einsum("ij,ija->iaj", fr, dx1x2_div_l2)
        # DONE: Don't reshape the tensor, reshape it in later process


class Deriv2Matern52(BaseKernel):
    def __init__(self, overall_scaling: torch.tensor, character_length: torch.tensor):
        super().__init__(overall_scaling, character_length)

    def forward(self, X1, X2: torch.tensor=None):
        device = X1.device
        DX1X2 = self._distance(X1, X2) 
        X1_l, X2_l = self._streched(X1, X2)
        r = self._r(X1_l, X2_l)
        sqrt_5 = math.sqrt(5)

        fr = (5.0/3.0)*torch.exp(-sqrt_5*r)  # n_sample * n_sample
        part1 = torch.einsum("ij,ab->iajb", (1+sqrt_5*r), (1.0/torch.pow(self.l, 2))*torch.eye(len(self.l), device=device))
        distance_div_l2 = torch.einsum("ija,ijb->iajb", DX1X2 / torch.pow(self.l, 2), DX1X2 / torch.pow(self.l, 2))
        k = torch.einsum("ij,iajb->iajb", fr, part1 - 5.0*distance_div_l2)
        return torch.pow(self.c, 2) * k
        # DONE: Same as above






