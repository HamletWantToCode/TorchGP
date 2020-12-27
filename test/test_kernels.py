import torch
from torch.autograd import grad
from gp import pairwise, Matern52, Deriv1Matern52, Deriv2Matern52
import unittest

torch.manual_seed(12345)

def test_basic_property(kernel, X1):
    K = kernel(X1, X1)
    is_symmetric = torch.allclose(K, K.T)
    res = torch.symeig(K)
    eigvs = res.eigenvalues
    is_pd = torch.all(eigvs > 0.0)
    return is_symmetric and is_pd

def test_gradient(kernel, d1kernel, X1, X2):
    X1.requires_grad_()
    K = kernel(X1, X2)
    ad_dldX1, = grad(K, X1, torch.ones_like(K))
    
    with torch.no_grad():
        dK1 = d1kernel(X1, X2)
        my_dldX1 = torch.einsum("ij,iaj->ia", torch.ones_like(K), dK1)
    
    isequal = torch.allclose(ad_dldX1, my_dldX1)
    return isequal

def test_hessian(d1kernel, d2kernel, X1, X2):
    X2.requires_grad_()
    dK1 = d1kernel(X1, X2)
    ad_d2l_dX1dX2, = grad(dK1, X2, torch.ones_like(dK1))

    with torch.no_grad():
        d2K = d2kernel(X1, X2)
        my_d2l_dX1dX2 = torch.einsum("iaj, iajb->jb", torch.ones_like(dK1), d2K)
    
    isequal = torch.allclose(ad_d2l_dX1dX2, my_d2l_dX1dX2)
    return isequal


class TestKernel(unittest.TestCase):
    def test_pairwise(self):
        x1 = torch.randn(1, 3)
        x2 = torch.randn(1, 3)
        d_x1x2 = pairwise(x1, x2)[0, 0]
        d_explicit_x1x2 = torch.sqrt(torch.sum(torch.pow(x1-x2, 2)))
        self.assertTrue(torch.allclose(d_x1x2, d_explicit_x1x2))

        d1_x1x1 = pairwise(x1, x1)[0, 0]
        self.assertTrue(torch.allclose(d1_x1x1, torch.zeros(1)))

    def test_basic(self):
        """
        1. shape of kernel matrix
        2. kernel PSD, symmetric
        """
        X1 = torch.randn(10, 3)
        l = torch.rand(3) * 10.0
        C = torch.tensor([3.0])

        matern = Matern52(C, l)
        self.assertTrue(test_basic_property(matern, X1))

        # since the return is a 4-tensor
        d2matern = Deriv2Matern52(C, l)
        d2K = d2matern(X1, X1)
        d2K = d2K.reshape(30, 30)
        self.assertTrue(torch.allclose(d2K, d2K.T))
        res = torch.symeig(d2K)
        eigvs = res.eigenvalues
        self.assertTrue(torch.all(eigvs > 0.0))

    def test_grad_hessian(self):
        X1 = torch.randn(4, 5)
        X2 = torch.randn(3, 5)
        l = torch.rand(5) * 10.0
        C = torch.tensor([2.0])

        matern = Matern52(C, l)
        dmatern = Deriv1Matern52(C, l)
        d2matern = Deriv2Matern52(C, l)
        self.assertTrue(test_gradient(matern, dmatern, X1, X2))
        self.assertTrue(test_hessian(dmatern, d2matern, X1, X2))





