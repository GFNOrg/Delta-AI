import torch
import os
import argparse

from utils.prob_models import IsingModel, FactorGraphModel
from algorithms.base_gfn import GFlowNet
from data import load_ising_data, load_factor_data
from utils.samplers_utils import linear_mmd
from utils.misc import Logger, makedir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)

    # data & model
    parser.add_argument('--model', type=str, default="ising", choices=["ising", "factor"])
    parser.add_argument('--graph', type=str, default="torus", choices=["ladder", "lattice", "torus"])
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--vdim', type=int, default=9, choices=[9, 81, 289, 1089])
    parser.add_argument('--hdim', type=int, default=512)

    # training
    parser.add_argument("--n_iters", type=int, default=100000)
    parser.add_argument('--batchsz', type=int, default=1000)
    parser.add_argument('--batchsz_mmd', type=int, default=1000)
    parser.add_argument("--q1lr", type=float, default=1e-3)
    parser.add_argument("--q2lr", type=float, default=1e-3)
    parser.add_argument("--zlr", type=float, default=1)

    # gfn
    parser.add_argument('--alg', type=str, default="tb", choices=["tb", "db", "subtb"])
    parser.add_argument("--uniform_pb", action="store_true")
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--rand_coef", type=float, default=0)
    parser.add_argument("--back_ratio", type=float, default=0)
    parser.add_argument("--forward_looking", action="store_true")
    parser.add_argument("--lamda", type=float, default=1)
    parser.add_argument('--mc_size', type=int, default=5)

    args = parser.parse_args()
    if args.vdim == 64:
        args.batchsz = 1000
        args.back_ratio = 0.4
        args.rand_coef = 0.5
        args.lamda = 1.17
    elif args.vdim == 256:
        args.batchsz = 500
        args.back_ratio = 0.2
        args.rand_coef = 0.1
        args.lamda = 1.04
    else:
        args.batchsz = 200
        args.back_ratio = 0.1
        args.rand_coef = 0.05
        args.lamda = 1.01

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
        path = f"vdim{args.vdim}_sigma{args.sigma}"
        J, b, samp_train, samp_eval = load_ising_data(args.graph, path, device)
        model = IsingModel(J, b).to(device)
    else:
        path = f"vdim{args.vdim}_sigma{args.sigma}_K{args.K}"
        W1, b1, W2, b2, samp_train, samp_eval = load_factor_data(args.graph, path, device)
        model = FactorGraphModel(args.graph, W1, b1, W2, b2).to(device)

    gfn = GFlowNet(args, device, model).to(device)

    param_list = [{'params': gfn.Qnet1.parameters(), 'lr': args.q1lr},
                  {'params': gfn.Qnet2.parameters(), 'lr': args.q2lr},
                  {'params': gfn.logZ, 'lr': args.zlr}]
    optimizer = torch.optim.Adam(param_list)

    T = lambda t: int(args.n_iters * t)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [T(0.5)], gamma=0.1)

    path_txt = os.path.join(args.save_dir, args.exp_name, "results.txt")
    if os.path.isfile(path_txt):
        os.remove(path_txt)

    logger.start()

    for i in range(args.n_iters + 1):
        batch = samp_train[torch.randperm(samp_train.shape[0])][:args.batchsz]
        loss_gfn = gfn.get_loss(batch)
        logger.meter("train", "loss_gfn", loss_gfn)

        if i % args.print_every == 0:
            nll = gfn.get_NLL(samp_eval, args.mc_size)
            logger.meter("train", "NLL", nll)

            V_samp = gfn.get_samples(args.batchsz_mmd)
            mmd = linear_mmd(samp_eval, V_samp)
            logger.meter("train", "MMD", mmd)

            with open(path_txt, "a") as f:
                f.write(f"{i} {nll.item()} {mmd}\n")

        logger.step()

        optimizer.zero_grad()
        loss_gfn.backward()
        optimizer.step()
        scheduler.step()

        if i % args.save_every == 0:
            path_prev = os.path.join(args.save_dir, args.exp_name, f"gfn_step{i - args.save_every}.pt")
            if os.path.isfile(path_prev):
                os.remove(path_prev)
            torch.save(gfn.state_dict(), os.path.join(args.save_dir, args.exp_name, f"gfn_step{i}.pt"))

    logger.finish()
