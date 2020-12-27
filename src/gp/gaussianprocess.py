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

    def forward(self, Xtrain: torch.tensor):
        """
        Xtrain: n_samples * n_features
        When output is a vector, it will be flattened into a long vector 
        """
        device = Xtrain.device
        
        K = self.kernel(Xtrain, Xtrain)
        if K.ndim == 4:
            n_samples, n_features = Xtrain.shape
            K = K.reshape((n_samples*n_features, n_samples*n_features))
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

        K = self.kernel(Xtrain, Xtrain)
        if K.ndim == 4:
            n_samples, n_features = Xtrain.shape
            K = K.reshape((n_samples*n_features, n_samples*n_features))
        Kprime = K + self.alpha * torch.eye(K.shape[0], device=device)
        L = torch.cholesky(Kprime)

        kXnew = self.kernel(Xnew, Xtrain)
        if kXnew.ndim == 4:
            n_newsamples, _ = Xnew.shape
            kXnew = kXnew.reshape((n_newsamples*n_features, n_samples*n_features))
            ytrain = ytrain.flatten()
        mean_ynew = self.mean(Xnew) + torch.matmul(kXnew, torch.cholesky_solve(ytrain[:, None], L))
        # LL^T _x = kXnew^T
        _x = torch.cholesky_solve(kXnew, L)
        
        kXnewXnew = self.kernel(Xnew, Xnew)
        if kXnewXnew.ndim == 4:
            kXnewXnew = kXnewXnew.reshape((n_newsamples*n_features, n_newsamples*n_features))
        std_ynew = torch.sqrt(torch.diag(kXnewXnew) - torch.einsum("ij,ji->i", kXnew, _x))
        return [Normal(mu, sigma) for mu,sigma in zip(mean_ynew, std_ynew)]

        




        


