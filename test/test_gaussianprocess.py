import torch
from gp import GaussianProcess, ZeroScalarMean, ZeroVectorMean, Matern52, Deriv2Matern52
import unittest
import logging

# logging.basicConfig(level=logging.DEBUG)

def basic_test(gaussprocess, X1, y1, X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    out_d = y1.shape[1]
    margin = gaussprocess(X1)
    type_check = isinstance(margin, torch.distributions.MultivariateNormal)
    logging.debug("basic test: margin typecheck={}".format(type_check))
    sample = margin.sample([1])
    sample_shape_check = (sample.shape==(1, n1*out_d))
    logging.debug("basic test: sample shape check={}".format(sample_shape_check))

    p_y2 = gaussprocess.predict(X2, X1, y1)
    type_check = type_check and isinstance(p_y2[0], torch.distributions.Normal)
    logging.debug("basic test: predict typecheck={}".format(type_check))

    len_check = (len(predict)==n2*out_d)
    logging.debug("basic test: lencheck={}".format(len_check))
    return type_check and sample_shape_check and len_check


class TestGP(unittest.TestCase):
    def test_basic(self):
        """
        1. GP forward output is a mvNormal
        2. GP predict output is a set of normal distributions
        3. sample shape
        """
        C = torch.tensor([3.0])
        l = torch.rand(3)
        X1 = torch.randn(10, 3)
        X2 = torch.randn(2, 3)
        y1 = torch.randn(10, 1)
        y1prime = torch.randn(10, 3)
        
        matern = Matern52(C, l)
        gp1 = GaussianProcess(ZeroScalarMean(), matern, 0.1)
        self.assertTrue(basic_test(gp1, X1, y1, X2))
        d2matern = Deriv2Matern52(C, l)
        gp2 = GaussianProcess(ZeroVectorMean(), d2matern, 0.1)
        self.assertTrue(basic_test(gp2, X1, y1prime, X2))
        
    def test_statistics(self):
        """
        1. sample from GP and calculate mean and covariance
        """
        C = torch.tensor([3.0])
        l = torch.rand(3)
        X1 = torch.randn(10, 3)
        matern = Matern52(C, l)
        gp1 = GaussianProcess(ZeroScalarMean(), matern, 0.0)
        margin = gp1(X1)
        n_sample = 100000
        samples = margin.sample([n_sample])

        statistic_mean = torch.mean(samples, dim=0)
        logging.debug("mean:{}".format(statistic_mean))
        self.assertTrue(torch.allclose(statistic_mean, torch.zeros(10), atol=0.1))
        statistic_cov = samples.T @ samples / (n_sample-1)
        K = matern(X1)
        logging.debug("Cov:{}".format(statistic_cov))
        self.assertTrue(torch.allclose(K, statistic_cov, atol=0.1))


