import torch
import os
import argparse
import math

from utils.prob_models import IsingModel, FactorGraphModel
from utils.pgm import PGM
from algorithms.order_gfn import GFlowNet
from data import load_ising_data, load_factor_data
from utils.misc import Logger, makedir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./results_new")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10000)

    # data & model
    parser.add_argument('--alg', type=str, default="tb", choices=["tb", "db", "dbfl"])
    parser.add_argument('--model', type=str, default="ising", choices=["ising", "factor"])
    parser.add_argument('--graph', type=str, default="lattice", choices=["ladder", "lattice"])
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=0.2)
    parser.add_argument('--vdim', type=int, default=64, choices=[64, 256, 1024])
    parser.add_argument('--hdim', type=int, default=512)

    # training
    parser.add_argument('--n_iters', type=int, default=200000)
    parser.add_argument('--batchsz', type=int, default=1000)
    parser.add_argument('--batchsz_nll', type=int, default=100)
    parser.add_argument('--glr', type=float, default=1e-3)
    parser.add_argument('--mlr', type=float, default=1e-1)
    parser.add_argument('--zlr', type=float, default=1e-1)

    # on/off policy
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--uniform', action="store_true")
    parser.add_argument('--back_ratio', type=float, default=0)

    # gfn
    parser.add_argument('--sample_dag_every', type=int, default=50)
    parser.add_argument('--n_nodes', type=int, default=1)
    parser.add_argument('--n_subdags_per_node', type=int, default=1)

    args = parser.parse_args()
    if args.n_nodes > args.vdim:
        raise AssertionError("n_nodes should be smaller than vdim")
    if args.model == "ising":
        args.sigma = 0.2
    else:
        args.sigma = 0.5

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
    gfn = GFlowNet(args, device, model, pgm).to(device)

    param_list = [
        {"params": [v for n, v in gfn.Qnet.named_parameters() if "marginals" in n],
            "lr": args.mlr},
        {"params": [v for n, v in gfn.Qnet.named_parameters() if "marginals" not in n],
            "lr": args.glr},
        {"params": gfn.Fnet.parameters(), "lr": args.glr},
        {'params': gfn.logZ, 'lr': args.zlr}
    ]
    optimizer = torch.optim.Adam(param_list)

    T = lambda x: int(args.n_iters * x)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [T(0.2), T(0.4), T(0.6), T(0.8), T(0.9)], gamma=0.1)

    path_txt = os.path.join(args.save_dir, args.exp_name, "nll.txt")
    if os.path.isfile(path_txt):
        os.remove(path_txt)

    logger.start()

    t = 0
    for i in range(args.n_iters + 1):
        if i % args.sample_dag_every == 0:
            pgm.sample_and_set_fulldag()

        V, loss = gfn.sampleV(temp=args.temp, epsilon=args.epsilon, uniform=args.uniform)

        if args.back_ratio > 0:
            batch = samp_train[torch.randperm(samp_train.shape[0])][:args.batchsz]
            _, data_loss = gfn.sampleV(temp=args.temp, epsilon=args.epsilon, uniform=args.uniform, data=batch)
            loss = (1-args.back_ratio) * loss + args.back_ratio * data_loss

        logger.meter("train", "loss_gfn", loss)

        if i == int(math.pow(2, t)):
            K_pa, _, _ = pgm.sample_fulldag()
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
