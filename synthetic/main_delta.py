import torch
import os
import argparse
import math

from utils.prob_models import IsingModel, FactorGraphModel
from utils.pgm import PGM
from utils.samplers_utils import linear_mmd
from algorithms.delta import DeltaAI
from data import load_ising_data, load_factor_data
from utils.misc import Logger, makedir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./results_new")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)

    # data & model
    parser.add_argument('--model', type=str, default="ising", choices=["ising", "factor"])
    parser.add_argument('--graph', type=str, default="lattice", choices=["ladder", "lattice", "torus"])
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--vdim', type=int, default=64)
    parser.add_argument('--hdim', type=int, default=512)
    parser.add_argument('--act', type=str, default='relu', choices=['relu', 'elu', 'tanh'])

    # training
    parser.add_argument('--n_iters', type=int, default=200000)
    parser.add_argument('--batchsz', type=int, default=1000)
    parser.add_argument('--batchsz_nll', type=int, default=100)
    parser.add_argument('--batchsz_mmd', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--marg_lr', type=float, default=1e-1)

    # exploration policy
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--uniform', action="store_true")

    # gfn
    parser.add_argument('--alg', type=str, default="rand", choices=["fixed", "rand"])
    parser.add_argument('--sample_subdags_every', type=int, default=50)
    parser.add_argument('--n_nodes', type=int, default=64)
    parser.add_argument('--n_subdags_per_node', type=int, default=1)

    # vs mcmc
    parser.add_argument('--vs_mcmc', action="store_true")
    parser.add_argument('--factor_steps', type=int, default=200)
    parser.add_argument('--sampleV_every', type=int, default=50)

    args = parser.parse_args()
    if args.alg == "rand":
        args.n_nodes = args.vdim
        args.n_subdags_per_node = args.batchsz // args.n_nodes + 1
        args.batchsz = args.n_nodes * args.n_subdags_per_node

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
    elif args.model == "factor":
        path = f"vdim{args.vdim}_sigma{args.sigma}_K{args.K}"
        W1, b1, W2, b2, samp_train, samp_eval = load_factor_data(args.graph, path, device)
        model = FactorGraphModel(args.graph, W1, b1, W2, b2).to(device)
    else:
        raise NotImplementedError

    adj = model.get_adj().to(device)
    pgm = PGM(args, device, adj).to(device)
    gfn = DeltaAI(args, device, model, pgm).to(device)

    param_list = [
        {"params": [v for n, v in gfn.Qnet.named_parameters() if "marginals" in n],
            "lr": args.marg_lr},
        {"params": [v for n, v in gfn.Qnet.named_parameters() if "marginals" not in n],
            "lr": args.lr}
    ]
    optimizer = torch.optim.Adam(param_list)

    T = lambda t: int(args.n_iters * t)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [T(0.2), T(0.4), T(0.6), T(0.8), T(0.9)], gamma=0.1)

    path_txt = os.path.join(args.save_dir, args.exp_name, "mmd.txt" if args.vs_mcmc else "nll.txt")
    if os.path.isfile(path_txt):
        os.remove(path_txt)

    pgm.sample_and_set_fulldag()

    logger.start()

    t = 0
    for i in range(args.n_iters + 1):

        if args.vs_mcmc:
            gfn.set_factor(min(1, i/args.factor_steps))

        if args.alg == "rand" or (args.alg == "fixed" and i % args.sampleV_every == 0):
            with torch.no_grad():
                if args.alg == "fixed":
                    ilist = torch.randint(args.vdim, (args.batchsz,)).to(device)

                elif args.alg == "rand":
                    if i % args.sample_subdags_every == 0:
                        ilist = torch.multinomial(torch.ones(args.vdim),
                                num_samples=args.n_nodes, replacement=False).to(device)
                        pgm.sample_and_set_subdags(ilist)
                        ilist = ilist.repeat_interleave(args.n_subdags_per_node, dim=0)

                if args.uniform and args.epsilon == 1:
                    p = 0.5 * torch.ones(args.batchsz, args.vdim).to(device)
                    V = 2 * torch.bernoulli(p) - 1
                else:
                    temp = max(1, args.temp - (args.temp-1)/args.factor_steps*i)
                    V = gfn.sampleV(args.batchsz, temp=temp, epsilon=args.epsilon, uniform=args.uniform)

        loss = gfn.get_loss(V, ilist)
        logger.meter("train", "loss", loss)

        if args.vs_mcmc:
            if i % args.print_every == 0:
                if args.alg == "rand":
                    K_pa, _, _ = pgm.sample_fulldag()
                else:
                    K_pa = pgm.K_pa_full
                V_samp = gfn.sampleV(args.batchsz, temp=1, epsilon=0, uniform=False)
                mmd = linear_mmd(samp_eval, V_samp)
                logger.meter("train", "MMD", mmd)
                with open(path_txt, "a") as f:
                    f.write(f"{i} {mmd}\n")
        else:
            if i == int(math.pow(2, t)):
                if args.alg == "rand":
                    K_pa, _, _ = pgm.sample_fulldag()
                else:
                    K_pa = pgm.K_pa_full
                nll = gfn.get_NLL(samp_eval, K_pa)
                logger.meter("train", "NLL", nll)
                with open(path_txt, "a") as f:
                    f.write(f"{i} {nll}\n")
                t += 1

        logger.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    logger.finish()
