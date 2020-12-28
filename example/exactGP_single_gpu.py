import math
import torch
import gp 
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt 
import logging

logging.basicConfig(level=logging.WARN)

train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2*math.pi)) + torch.randn(train_x.size()) * 0.2
train_x = train_x[:, None]

C = torch.tensor([1.0])
l = torch.rand(1)
rbf = gp.RBF(C, l)
gaussprocess = gp.GaussianProcess(gp.ZeroScalarMean(), rbf, 2e-2)

train_x = train_x.cuda()
train_y = train_y.cuda()
gaussprocess = gaussprocess.cuda()

optimizer = torch.optim.Adam(gaussprocess.parameters(), lr=0.1)
training_iter = 50
with tqdm(total=training_iter) as pbar:
    for i in range(training_iter):
        optimizer.zero_grad()
        margin = gaussprocess(train_x)
        nll = -margin.log_prob(train_y)
        nll.backward()

        pbar.set_postfix(
            loss="{:.6f}".format(nll.item()),
            length_scale="{:.4f}".format(gaussprocess.kernel.l[0].item()),
            prefactor="{:.4f}".format(gaussprocess.kernel.c[0].item())
        )

        optimizer.step()
        pbar.update(1)


test_x = torch.linspace(0, 1, 51)[:, None].cuda()
with torch.no_grad():
    p_testy = gaussprocess.predict(test_x, train_x, train_y)
    test_y_mean = torch.tensor([p_testy[i].mean for i in range(len(p_testy))], device=test_x.device)
    test_y_var = torch.tensor([p_testy[i].stddev for i in range(len(p_testy))], device=test_x.device)

test_y_mean = test_y_mean.cpu()
test_y_var = test_y_var.cpu()
train_x = train_x.cpu().flatten()
train_y = train_y.cpu()
test_x = test_x.cpu().flatten()

train_data = torch.stack((train_x, train_y))
test_data = torch.stack([test_x, test_y_mean, test_y_var])
torch.save(train_data, "train.pt")
torch.save(test_data, "test.pt")

logging.debug("mean shape={}".format(test_y_mean.shape))
logging.debug("var shape={}".format(test_y_var.shape))


# with torch.no_grad():
#     f, ax = plt.subplots(1, 1, figsize=(4, 3))
#     ax.plot(train_x.numpy(), train_y.numpy(), "k*")
#     ax.plot(test_x.numpy(), test_y_mean.numpy(), "b")
#     ax.fill_between(test_x.numpy(), test_y_var.numpy(), test_y_var.numpy(), alpha=0.5)
#     ax.set_ylim([-3, 3])
#     ax.legend(["observed data", "mean", "confidence"])
#     f.savefig("sin.png")


