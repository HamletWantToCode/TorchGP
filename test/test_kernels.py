import torch
from torch.autograd.gradcheck import get_numerical_jacobian
from gp import Matern52, Deriv1Matern52, Deriv2Matern52
import unittest
import logging
# logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(12345)

def test_basic_property(K: torch.tensor):
    is_symmetric = torch.allclose(K, K.T)
    logging.debug("test basic: issymmetric={}".format(is_symmetric))
    res = torch.symeig(K)
    eigvs = res.eigenvalues
    is_pd = (torch.all(eigvs > 0.0)).item()
    logging.debug("test basic: ispositivedefinite={}".format(is_pd))
    return is_symmetric and is_pd

def test_gradient(kernel, d1kernel, X1, X2):
    """
    X1: 1 * n_features
    X2: n_samples * n_features
    """
    n_samples, n_features = X2.shape
    X1.requires_grad_()
    # # test gradient for single input
    # fd_J = get_numerical_jacobian(lambda x: kernel(x), X1, eps=1e-5)
    # with torch.no_grad():
    #     my_J = d1kernel(X1)
    #     my_J = my_J.reshape((n_features, 1))
    # isequal = torch.allclose(fd_J, my_J)

    # test gradient for two inputs
    fd_J1 = get_numerical_jacobian(lambda x: kernel(x, X2), X1, eps=1e-5)
    with torch.no_grad():
        my_J1 = d1kernel(X1, X2)
        my_J1 = my_J1.reshape((n_features, n_samples))
    isequal = torch.allclose(fd_J1, my_J1)
    return isequal

def test_hessian(d1kernel, d2kernel, X1, X2):
    """
    X1: 1 * n_features
    X2: 1 * n_features
    """
    _, n_features = X1.shape
    X2.requires_grad_()
    # WARN: why manurally computed hessian is not zero on it's diagonal?
    # DONE: This is not a bug, refer to https://docs.gpytorch.ai/en/stable/_modules/gpytorch/kernels/rbf_kernel_grad.html#RBFKernelGrad. We will only test the hessian w.r.t different inputs (x1 != x2)
    # fd_H = get_numerical_jacobian(lambda x: d1kernel(x), X2, eps=1e-5)
    # with torch.no_grad():
    #     my_H = d2kernel(X2)
    #     my_H = my_H.reshape((n_features, n_features))
    # isequal = torch.allclose(fd_H, my_H)

    # test hessian for two inputs
    fd_H1 = get_numerical_jacobian(lambda x: d1kernel(X1, x), X2, eps=1e-5)
    with torch.no_grad():
        my_H1 = d2kernel(X1, X2)
        my_H1 = my_H1.reshape((n_features, n_features))
    isequal = torch.allclose(fd_H1, my_H1)
    return isequal


class TestKernel(unittest.TestCase):
    def test_basic(self):
        """
        1. shape of kernel matrix
        2. kernel PSD, symmetric
        """
        X1 = torch.randn(10, 3)
        l = torch.rand(3) * 10.0
        C = torch.tensor([3.0])

        matern = Matern52(C, l)
        K1 = matern(X1)
        self.assertTrue(test_basic_property(K1))

        # since the return is a 4-tensor
        d2matern = Deriv2Matern52(C, l)
        K2 = d2matern(X1)
        K2 = K2.reshape((30, 30))
        self.assertTrue(test_basic_property(K2))

        # check consistency
        K3 = matern(X1, X1)
        self.assertTrue(torch.allclose(K1, K3))
        K4 = d2matern(X1, X1)
        K4 = K4.reshape((30, 30))
        self.assertTrue(torch.allclose(K2, K4))

    def test_deriv_kernel(self):
        X1 = torch.randn(1, 5).double()
        X2 = torch.randn(3, 5).double()
        X3 = torch.randn(1, 5).double()
        l = torch.rand(5) * 10.0
        C = torch.tensor([2.0])

        matern = Matern52(C, l).double()
        dmatern = Deriv1Matern52(C, l).double()
        d2matern = Deriv2Matern52(C, l).double()
        self.assertTrue(test_gradient(matern, dmatern, X1, X2))
        self.assertTrue(test_hessian(dmatern, d2matern, X1, X3))



