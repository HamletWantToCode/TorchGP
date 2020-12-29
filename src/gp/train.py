import torch
import logging
from tqdm.autonotebook import tqdm
from .utils import get_mean_and_var
from torch.utils.tensorboard import SummaryWriter

def default_callback(gaussprocess, writer, step):
    for i in range(len(gaussprocess.kernel.l)):
        writer.add_scalar("Hyperparameter/LengthScale/l{:d}".format(i), gaussprocess.kernel.l[i], step)
    writer.add_scalar("Hyperparameter/lambda^2", torch.pow(gaussprocess.kernel.c, 2)[0], step)

def train(
    train_data: tuple,
    gaussprocess,
    opt,
    num_epoch,
    device,
    workdir=None,
    callback=default_callback
    ):

    if workdir:
        writer = SummaryWriter(log_dir=workdir)
    else:
        writer = SummaryWriter()

    train_X, train_Y = train_data
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    gaussprocess.to(device)
    state = {}
    with tqdm(total=num_epoch) as pbar:
        pbar.set_description("Train")
        logging.info("Training......")
        for epoch_ix in range(num_epoch):
            opt.zero_grad()

            marginal = gaussprocess(train_X)
            negloglik = -1 * marginal.log_prob(train_Y.flatten())

            state.update({"negloglik": "{:.10f}".format(negloglik)})
            if callback:
                callback(gaussprocess, writer, epoch_ix)
            pbar.set_postfix(negloglik=state["negloglik"])
            pbar.update(1)

            negloglik.backward()
            if sum(torch.sum(torch.isnan(p.grad)) for p in gaussprocess.parameters()) == 0:
                gnorm = 0.0
                for p in gaussprocess.parameters():
                    gnorm += torch.norm(p.grad)
                state.update({"gnorm": "{:.10f}".format(gnorm)})
                opt.step()
            else:
                logging.warn("Catch NaN value in gradient!")

            writer.add_scalar("Train/negloglik", negloglik, epoch_ix)
            log_str = ["%s=%s" %(k, v) for k,v in state.items()]
            log_str = ", ".join(log_str)
            logging.info(log_str)


def evaluate(test_X, train_data: tuple, gaussprocess, device, output_shape: tuple):
    test_X = test_X.to(device)
    train_X, train_Y = train_data
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    gaussprocess.to(device)
    with torch.no_grad():
        p_testY = gaussprocess.predict(test_X, *train_data)
        mean_testY, std_testY = get_mean_and_var(p_testY, output_shape)   # Now, data is on CPU
    return mean_testY, std_testY


