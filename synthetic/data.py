import torch
import math
import argparse
import os

from utils.prob_models import IsingModel, FactorGraphModel
from utils.gibbs import get_gibbs_sample
from utils.misc import makedir

def load_ising_data(graph, path, device):
    path = os.path.join("data", f"ising_{graph}", path)
    J = torch.load(os.path.join(path, "J.pt")).float().to(device)
    b = torch.load(os.path.join(path, "b.pt")).float().to(device)
    samp_train = torch.load(os.path.join(path, "samp_train.pt")).float().to(device)
    samp_eval = torch.load(os.path.join(path, "samp_eval.pt")).float().to(device)
    return J, b, samp_train, samp_eval

def load_factor_data(graph, path, device):
    path = os.path.join("data", f"factor_{graph}", path)
    W1 = torch.load(os.path.join(path, "W1.pt")).float().to(device)
    b1 = torch.load(os.path.join(path, "b1.pt")).float().to(device)
    W2 = torch.load(os.path.join(path, "W2.pt")).float().to(device)
    b2 = torch.load(os.path.join(path, "b2.pt")).float().to(device)
    samp_train = torch.load(os.path.join(path, "samp_train.pt")).float().to(device)
    samp_eval = torch.load(os.path.join(path, "samp_eval.pt")).float().to(device)
    return W1, b1, W2, b2, samp_train, samp_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="ising", choices=["ising", "factor"])
    parser.add_argument('--graph', type=str, default="torus", choices=["ladder", "lattice", 'torus'])
    parser.add_argument('--vdim', type=int, default=64, choices=[64, 256, 512, 1024])
    parser.add_argument('--n_samp', type=int, default=20000)
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--n_annealing_steps', type=int, default=10000)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--hdim', type=int, default=10)
    parser.add_argument('--K', type=int, default=4)
    args = parser.parse_args()

    vdim = args.vdim

    if args.model == "ising":
        if args.graph == "ladder":
            J = torch.zeros(vdim, vdim)
            for i in range(vdim-1):
                J[i+1, i] = 1.
                if i+2 < vdim:
                    J[i+2, i] = 1.

        elif args.graph in ["lattice", "torus"]:
            grid_len = int(math.sqrt(vdim))

            if vdim % grid_len != 0:
                raise ValueError("vdim should be a perfect square")

            J = torch.zeros(vdim, vdim)
            for i in range(grid_len):
                for j in range(grid_len):
                    if i + 1 < grid_len:
                        J[i * grid_len + j, (i + 1) * grid_len + j] = 1.
                    if j + 1 < grid_len:
                        J[i * grid_len + j, i * grid_len + j + 1] = 1.
                    if args.graph == "torus":
                        if i == 0:
                            J[j, (grid_len-1)*grid_len + j] = 1.
                        if j == 0:
                            J[i * grid_len, i * grid_len + grid_len-1] = 1.

        p = 0.5*torch.ones_like(J)
        J = J * (2*torch.bernoulli(p)-1)
        J = J + J.t()
        J *= args.sigma

        p = 0.5*torch.ones(vdim)
        b = 2*torch.bernoulli(p)-1
        b *= args.sigma

        model = IsingModel(J.cuda(), b.cuda())
        samp = get_gibbs_sample(model, args.n_samp, args.n_steps, args.n_annealing_steps)

        path = os.path.join("data", f"{args.model}_{args.graph}")
        makedir(path)
        path = os.path.join(path, f"vdim{args.vdim}_+-sigma{args.sigma}")
        makedir(path)

        torch.save(J, os.path.join(path, "J.pt"))
        torch.save(b, os.path.join(path, "b.pt"))
        torch.save(samp[:10000], os.path.join(path, "samp_eval.pt"))
        torch.save(samp[10000:], os.path.join(path, "samp_train.pt"))

    elif args.model == "factor":
        assert args.K == 4
        if args.graph == "ladder":
            n_factors = (args.vdim - 2) // 2
        else:
            n_factors = (int(math.sqrt(args.vdim)) - 1)**2

        W1 = args.sigma * torch.randn(n_factors, args.K, args.hdim)
        b1 = args.sigma * torch.randn(n_factors, 1, args.hdim)
        W2 = args.sigma * torch.randn(n_factors, args.hdim, 1)
        b2 = args.sigma * torch.randn(n_factors, 1, 1)

        model = FactorGraphModel(args.graph, W1.cuda(), b1.cuda(), W2.cuda(), b2.cuda())
        samp = get_gibbs_sample(model, args.n_samp, args.n_steps)

        path = os.path.join("data", f"{args.model}_{args.graph}")
        makedir(path)
        path = os.path.join(path, f"vdim{args.vdim}_sigma{args.sigma}_K{args.K}")
        makedir(path)

        torch.save(W1, os.path.join(path, "W1.pt"))
        torch.save(b1, os.path.join(path, "b1.pt"))
        torch.save(W2, os.path.join(path, "W2.pt"))
        torch.save(b2, os.path.join(path, "b2.pt"))
        torch.save(samp[:10000], os.path.join(path, "samp_eval.pt"))
        torch.save(samp[10000:], os.path.join(path, "samp_train.pt"))

    else:
        raise NotImplementedError
