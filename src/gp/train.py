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
    optimizer,
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

    def closure():
        optimizer.zero_grad()
        marginal = gaussprocess(train_X)
        negloglik = -1 * marginal.log_prob(train_Y.flatten())
        negloglik.backward()
        return negloglik

    with tqdm(total=num_epoch) as pbar:
        pbar.set_description("Train")
        logging.info("Training......")
        for epoch_ix in range(num_epoch):
            if callback:
                callback(gaussprocess, writer, epoch_ix)
            
            negloglik = optimizer.step(closure)

            state.update({"negloglik": "{:.10f}".format(negloglik)})
            pbar.set_postfix(negloglik=state["negloglik"])
            pbar.update(1)

            gnorm = 0.0
            for p in gaussprocess.parameters():
                gnorm += torch.norm(p.grad)
            state.update({"gnorm": "{:.10f}".format(gnorm)})

            writer.add_scalar("Train/negloglik", negloglik, epoch_ix)
            log_str = ["%s=%s" %(k, v) for k,v in state.items()]
            log_str = ", ".join(log_str)
            logging.info(log_str)


def evaluate(valid_data: tuple, train_data: tuple, gaussprocess, measure, device, returnY=False):
    valid_X, valid_Y = valid_data
    train_X, train_Y = train_data
    valid_X, valid_Y = valid_X.to(device), valid_Y.to(device)
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    gaussprocess.to(device)
    with torch.no_grad():
        p_testY = gaussprocess.predict(valid_X, train_X, train_Y)
        mean_validY, std_validY = get_mean_and_var(p_testY, valid_Y.shape, device)   # Now, data is on CPU
    if returnY:
        return (mean_validY, std_validY), measure(mean_validY, valid_Y)
    else:
        return measure(mean_validY, valid_Y)


