import torch
import numpy as np
import os

"""
for GWG
"""
def difference_function(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        x_pert = x.clone()
        x_pert[:, i] = 1. - x[:, i]
        delta = model(x_pert).squeeze() - orig_out
        d[:, i] = delta
    return d

def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()

def difference_function_multi_dim(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        for j in range(x.size(2)):
            x_pert = x.clone()
            x_pert[:, i] = 0.
            x_pert[:, i, j] = 1.
            delta = model(x_pert).squeeze() - orig_out
            d[:, i, j] = delta
    return d

def approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur

def short_run_mcmc(logp_net, x_init, k, sigma, step_size=None):
    x_k = torch.autograd.Variable(x_init, requires_grad=True)
    # sgld
    if step_size is None:
        step_size = (sigma ** 2.) / 2.
    for i in range(k):
        f_prime = torch.autograd.grad(logp_net(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += step_size * f_prime + sigma * torch.randn_like(x_k)

    return x_k

def l1(module):
    loss = 0.
    for p in module.parameters():
        loss += p.abs().sum()
    return loss

def get_ess(chain, burn_in):
    import tensorflow_probability as tfp
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv

def get_log_rmse(x):
    x = 2. * x - 1.
    x2 = (x ** 2).mean(-1)
    return x2.log10().detach().cpu().numpy()

def linear_mmd(x, y):
    x = x.float()
    y = y.float()

    with torch.no_grad():
        kxx = torch.mm(x, x.transpose(0, 1))
        idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
        kxx = kxx * (1 - torch.eye(x.shape[0]).to(x.device))
        kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

        kyy = torch.mm(y, y.transpose(0, 1))
        idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
        kyy[idx, idx] = 0.0
        kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
        kxy = torch.sum(torch.mm(y, x.transpose(0, 1))) / x.shape[0] / y.shape[0]
        mmd = kxx + kyy - 2 * kxy
    return mmd
