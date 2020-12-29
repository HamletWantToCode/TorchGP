import torch
import logging

def pairwise1(X1: torch.tensor):
    """
    X1: n_samples * n_features
    """
    device = X1.device
    _X1 = X1[:, None]
    pw_diff = _X1 - X1
    dsquare = torch.sum(torch.pow(pw_diff, 2), dim=-1)

    # TODO: check `gradchek`, why there will be None
    def replace_diag_with_zero(grad):
        if grad is not None:
            mask = torch.eye(*grad.shape, device=device).bool()
            grad.masked_fill_(mask, 0.0)
        else:
            logging.warn("gradient is None!!!")
        return grad

    if X1.requires_grad:
        dsquare.register_hook(replace_diag_with_zero)
    pw_distance = torch.sqrt(dsquare)
    return pw_distance

def pairwise2(X1: torch.tensor, X2: torch.tensor):
    """
    X1: n_samples * ... * n_feature
    X2: same as above
    NOTE: X1 != X1
    """
    _X1 = X1[:, None]
    pw_diff = _X1 - X2
    dsquare = torch.sum(torch.pow(pw_diff, 2), dim=-1)
    pw_distance = torch.sqrt(dsquare)
    return pw_distance

def get_mean_and_var(p_Y, output_shape: tuple):
    mean_Y = torch.tensor([p_Y[i].mean for i in range(len(p_Y))])
    mean_Y = mean_Y.reshape(output_shape)
    std_Y = torch.tensor([p_Y[i].stddev for i in range(len(p_Y))])
    std_Y = std_Y.reshape(output_shape)
    return mean_Y, std_Y
