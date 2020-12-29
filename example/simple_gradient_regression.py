"""
f(x, y) = \sin(x)\cos(y)
\frac{\partial f}{\partial x}=\cos(x)\cos(y)
\frac{\partial f}{\partial y}=-\sin(x)\sin(y)
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

x = torch.rand(100)
y = torch.rand(100)
train_X = torch.cat((x[:, None], y[:, None]), dim=-1)
dfdx = 2*math.pi*torch.cos((2*math.pi)*x) * torch.cos((2*math.pi)*y) + torch.randn(x.size()) * 0.2
dfdy = -2*math.pi*torch.sin((2*math.pi)*x) * torch.sin((2*math.pi)*y) + torch.randn(x.size()) * 0.2
train_Y = torch.cat((dfdx[:, None], dfdy[:, None]), dim=-1)

C = torch.tensor([5.0])
l = torch.rand(2)
kernel = gp.Deriv2Matern52(C, l)
gaussprocess = gp.GaussianProcess(gp.ZeroVectorMean(), kernel, 1e-2)
optimizer = torch.optim.Adam(gaussprocess.parameters(), lr=0.1)
training_iter = 50
gp.train((train_X, train_Y), gaussprocess, optimizer, training_iter, device, workdir="data/%s" %(current_time))

x1 = torch.ones(51) * 0.1
y1 = torch.linspace(0, 1, 51)
test_X = torch.cat((x1[:, None], y1[:, None]), dim=-1)
# test_xx, test_yy = torch.meshgrid(x1, y1)
# test_X = torch.cat((test_xx.flatten()[:, None], test_yy.flatten()[:, None]), dim=-1)
test_Y_mean, test_Y_var = gp.evaluate(test_X, (train_X, train_Y), gaussprocess, device, (51, 2))

train_X = train_X.cpu()
train_Y = train_Y.cpu()
test_X = test_X.cpu()

train_data = torch.cat((train_X, train_Y), dim=-1)
test_data = torch.cat([test_X, test_Y_mean, test_Y_var], dim=-1)
torch.save(train_data, "data/%s/train.pt" %(current_time))
torch.save(test_data, "data/%s/test.pt" %(current_time))

