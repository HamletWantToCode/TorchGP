import torch
import torch.nn as nn 
from torch.distributions import Normal, MultivariateNormal
from .kernels import BaseKernel
from .means import BaseMean

class GaussianProcess(nn.Module):
    def __init__(self, mean: BaseMean, kernel: BaseKernel, regularizer: torch.tensor):
        super().__init__()
        self.mean = mean
        self.kernel = kernel
        self.alpha = regularizer

    @staticmethod
    def to_matrix(K):
        if K.ndim == 2:
            return K
        elif K.ndim == 4:
            n1, d1, n2, d2 = K.shape
            return K.reshape((n1*d1, n2*d2))

    def forward(self, Xtrain: torch.tensor):
        """
        Xtrain: n_samples * n_features
        When output is a vector, it will be flattened into a long vector 
        """
        device = Xtrain.device
        
        K = self.to_matrix(self.kernel(Xtrain))
        Kprime = K + self.alpha * torch.eye(K.shape[0], device=device)
        L = torch.cholesky(Kprime)
        
        mean = self.mean(Xtrain)
        marginal = MultivariateNormal(mean, scale_tril=L)
        return marginal

    def predict(self, Xnew: torch.tensor, Xtrain: torch.tensor, ytrain: torch.tensor):
        """
        Xnew: n_newsamples * n_features
        Xtrain: n_samples * n_features
        ytrain: n_samples * n_output
        """
        device = Xnew.device

        K = self.to_matrix(self.kernel(Xtrain))
        Kprime = K + self.alpha * torch.eye(K.shape[0], device=device)
        L = torch.cholesky(Kprime)

        kXnew = self.to_matrix(self.kernel(Xnew, Xtrain))
        ytrain = ytrain.flatten()
        mean_ynew = self.mean(Xnew) + torch.matmul(kXnew, torch.cholesky_solve(ytrain[:, None], L))
        # LL^T _x = kXnew^T
        _x = torch.cholesky_solve(kXnew.T, L)
        
        kXnewXnew = self.to_matrix(self.kernel(Xnew))
        std_ynew = torch.sqrt(torch.diag(kXnewXnew) - torch.einsum("ij,ji->i", kXnew, _x))
        return [Normal(mu, sigma) for mu,sigma in zip(mean_ynew, std_ynew)]

        




        


