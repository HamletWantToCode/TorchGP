import torch
from torch.autograd import grad, gradcheck
from gp.utils import pairwise1, pairwise2
import unittest

class TestUtils(unittest.TestCase):
    def test_pairwise(self):
        X1 = torch.randn(4, 3)
        X2 = torch.randn(2, 3)
        d1 = pairwise1(X1)
        d2 = pairwise2(X1, X2)

        # shape
        self.assertTrue(d1.shape == (4, 4))
        self.assertTrue(d2.shape == (4, 2))

        # consistancy
        d1_ = pairwise2(X1, X1)
        self.assertTrue(torch.allclose(d1_, d1))

        # zero diag
        self.assertTrue(torch.allclose(torch.diag(d1), torch.zeros(4)))

        # verify with explicit calculation
        self.assertTrue(d1[1, 3] == torch.sqrt(torch.sum(torch.pow(X1[1, :]-X1[3, :], 2))))
        self.assertTrue(d2[1, 1] == torch.sqrt(torch.sum(torch.pow(X1[1, :]-X2[1, :], 2))))

    def test_backprop(self):
        X1 = torch.randn(4, 3).double()
        X2 = torch.randn(2, 3).double()
        X1.requires_grad_()
        X2.requires_grad_()

        self.assertTrue(gradcheck(pairwise1, X1))
        self.assertTrue(gradcheck(pairwise2, (X1, X2)))

        # grad will be `nan` if backprop through `pairwise2` when X1==X2
        D = pairwise2(X1, X1)
        X1_bar, = grad(D, X1, torch.ones_like(D))
        self.assertTrue(torch.all(torch.isnan(X1_bar)))





        