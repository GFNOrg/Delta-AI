import torch
import os
import argparse

from utils.prob_models import IsingModel, FactorGraphModel
from utils.samplers_utils import linear_mmd
from algorithms.samplers import get_sampler
from data import load_ising_data, load_factor_data
from utils.misc import Logger, makedir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./results_new")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=100)

    # data & model
    parser.add_argument('--model', type=str, default="ising", choices=["ising", "factor"])
    parser.add_argument('--graph', type=str, default="ladder", choices=["ladder", "lattice"])
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument('--vdim', type=int, default=64, choices=[64, 256, 1024])

    # algorithm
    parser.add_argument('--alg', type=str, default="gibbs", choices=["gibbs", "gwg"])
    parser.add_argument('--n_iters', type=int, default=1000000)
    parser.add_argument('--burnin', type=int, default=10000)
    parser.add_argument('--batchsz', type=int, default=1000)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.gpu)
    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda")

    makedir(args.save_dir)
    makedir(os.path.join(args.save_dir, args.exp_name))

    logger = Logger(
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        print_every=args.print_every,
        save_every=args.n_iters,
        total_step=args.n_iters,
        print_to_stdout=True,
        wandb_project_name="",
        wandb_tags=[],
        wandb_config=args,
    )

    if args.model == "ising":
        path = f"vdim{args.vdim}_+-sigma{args.sigma}"
        J, b, samp_train, samp_eval = load_ising_data(args.graph, path, device)
        model = IsingModel(J, b).to(device)
    else:
        path = f"vdim{args.vdim}_sigma{args.sigma}_K{args.K}"
        W1, b1, W2, b2, samp_train, samp_eval = load_factor_data(args.graph, path, device)
        model = FactorGraphModel(args.graph, W1, b1, W2, b2).to(device)

    sampler = get_sampler(args.alg, args.vdim)
    x = torch.randint(low=0, high=2, size=(args.batchsz, args.vdim)).float().to(device)

    path_txt = os.path.join(args.save_dir, args.exp_name, "mmd.txt")
    if os.path.isfile(path_txt):
        os.remove(path_txt)

    logger.start()

    for i in range(args.n_iters):
        x = sampler.step(x, model).detach() # batchsz, vdim
        if i % args.print_every == 0:
            mmd = linear_mmd(samp_train, x*2-1)
            logger.meter("train", "MMD", mmd)
            with open(path_txt, "a") as f:
                f.write(f"{i} {mmd}\n")
        logger.step()
    logger.finish()
