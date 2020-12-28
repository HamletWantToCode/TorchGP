import torch
from gp import GaussianProcess, ZeroScalarMean, ZeroVectorMean, Matern52, Deriv2Matern52
import unittest
import logging

# logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(1234)
gpu_device = torch.device("cuda:0")

def check_device(gaussprocess, X, Y, X1):
    new_n_sample = X1.shape[0]
    output_d = Y.shape[1]

    # cpu
    with torch.no_grad():
        margin = gaussprocess(X)
        loglik = margin.log_prob(Y.flatten())
        logging.debug("loglik(cpu)={}".format(loglik))
        p_Y1 = gaussprocess.predict(X1, X, Y)
        mean_Y1 = torch.tensor([p_Y1[i].mean for i in range(len(p_Y1))])
        mean_Y1 = mean_Y1.reshape((new_n_sample, output_d))
        logging.debug("mean_Y1(cpu)={}".format(mean_Y1))

    # gpu
    X_gpu = X.to(gpu_device)
    Y_gpu = Y.to(gpu_device)
    X1_gpu = X1.to(gpu_device)
    gaussprocess.to(gpu_device)
    with torch.no_grad():
        margin_gpu = gaussprocess(X_gpu)
        loglik_gpu = margin_gpu.log_prob(Y_gpu.flatten())
        logging.debug("loglik(gpu)={}".format(loglik_gpu))
        p_Y1_gpu = gaussprocess.predict(X1_gpu, X_gpu, Y_gpu)
        mean_Y1_gpu = torch.tensor([p_Y1[i].mean for i in range(len(p_Y1_gpu))], device=gpu_device)
        mean_Y1_gpu = mean_Y1_gpu.reshape((new_n_sample, output_d))
        logging.debug("mean_Y1(gpu)={}".format(mean_Y1_gpu))
    loglik_gpu2cpu = loglik_gpu.cpu()
    mean_Y1_gpu2cpu = mean_Y1_gpu.cpu()

    return torch.allclose(loglik, loglik_gpu2cpu) and torch.allclose(mean_Y1, mean_Y1_gpu2cpu)


class TestDevice(unittest.TestCase):
    def test_cpu_gpu(self):
        C = torch.tensor([1.0])
        l = torch.rand(5)
        X = torch.randn(10, 5)
        y = torch.randn(10, 1)
        Y = torch.randn(10, 5)
        X1 = torch.randn(2, 5)
        matern = Matern52(C, l)
        d2matern = Deriv2Matern52(C, l)
        gp1 = GaussianProcess(ZeroScalarMean(), matern, 1e-2)
        gp2 = GaussianProcess(ZeroVectorMean(), d2matern, 1e-2)

        self.assertTrue(check_device(gp1, X, y, X1))
        self.assertTrue(check_device(gp2, X, Y, X1))

        


