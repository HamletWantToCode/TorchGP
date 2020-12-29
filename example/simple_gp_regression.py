"""
y=\sin(x) + 0.2\epsilon, \text{where }\epsilon\sim\mathcal{N}(0, 1)
"""
import math
import torch
import gp 
from tqdm.autonotebook import tqdm
import logging
import time
from pathlib import Path

torch.manual_seed(1234)

current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
Path("data/%s" %(current_time)).mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename="data/%s/exactGP.log" %(current_time), level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2*math.pi)) + torch.randn(train_x.size()) * 0.2
train_x = train_x[:, None]

C = torch.tensor([1.0])
l = torch.rand(1)
kernel = gp.Matern52(C, l)
gaussprocess = gp.GaussianProcess(gp.ZeroScalarMean(), kernel, 2e-2)
optimizer = torch.optim.LBFGS(gaussprocess.parameters(), lr=0.1)
training_iter = 20

gp.train((train_x, train_y), gaussprocess, optimizer, training_iter, device, workdir="data/%s" %(current_time))
test_x = torch.linspace(0, 1, 51)
test_y = torch.sin(test_x * (2*math.pi))
test_x = test_x[:, None]
mse = torch.nn.MSELoss()
(test_y_mean, test_y_var), _ = gp.evaluate((test_x, test_y), (train_x, train_y), gaussprocess, mse, device, returnY=True)

train_x = train_x.cpu().flatten()
train_y = train_y.cpu()
test_x = test_x.cpu().flatten()
test_y_mean = test_y_mean.cpu()
test_y_var = test_y_var.cpu()

train_data = torch.stack((train_x, train_y))
test_data = torch.stack([test_x, test_y_mean, test_y_var])
torch.save(train_data, "data/%s/train.pt" %(current_time))
torch.save(test_data, "data/%s/test.pt" %(current_time))


