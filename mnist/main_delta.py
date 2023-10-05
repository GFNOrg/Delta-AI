import torch
import os
import argparse

from utils.pgm import PGM
from algorithms.delta import DeltaAI
from data import load_mnist, Pyramid
from utils.misc import makedir, get_NLL_importance
from algorithms.gibbs import GibbsSampler

def get_ilist_and_set_dags(args, pgm, p_ilist_F):
    # with torch.no_grad():
    if args.alg == "fixed":
        ilist = torch.multinomial(p_ilist_F, num_samples=args.n_nodes, replacement=False).to(
            args.device)

    elif args.alg == "rand":
        if args.sampling_dag == "partial":
            # Sample a new set of partial DAGs
            ilist = torch.multinomial(p_ilist_F, num_samples=args.n_nodes, replacement=False).to(
                args.device)
            pgm.sample_and_set_subdags(ilist)
            if args.n_subdags_per_node > 1:
                ilist = torch.repeat_interleave(ilist, args.n_subdags_per_node, dim=0)
        else:
            # Sample a new full DAG
            pgm.sample_and_set_fulldag(batchsz=args.n_fulldags_per_batch)
            ilist = torch.multinomial(p_ilist_F, num_samples=args.n_nodes, replacement=False).to(
                args.device)
    else:
        raise ValueError(f"Unknown algorithm {args.alg}")

    if args.repeat_nodes > 1:
        ilist = torch.repeat_interleave(ilist, args.repeat_nodes, dim=0)

    return ilist

def train_p(args, X_batch, q, p, optimizer_p, V_gibbs=None, p_gibbs=None):
    if args.q_objective == "gibbs":
        V_gibbs[:, :args.xdim] = X_batch
        V_gibbs[:, args.xdim:] = 2 * torch.bernoulli(p_gibbs) - 1
        with torch.no_grad():
            V_q = q.get_gibbs_from_data(V_gibbs, start_idx=args.xdim)

    else:
        V_q = q.sampleV(args.batchsz, args.sampling_dag, temp=1, epsilon=0, X=X_batch)

    loss = -p.probV(V_q, args.sampling_dag, log=True).mean()

    optimizer_p.zero_grad()
    loss.backward()
    optimizer_p.step()

    return {"train/loss_p": loss.item()}

def sleep(args, q, p):
    # Sleep training for q
    V_p = p.sampleV(args.batchsz, "full", temp=1, epsilon=0)  # on policy
    loss = -q.probV(V_p, "full", log=True).mean()

    return loss

def train_q(args, X_batch, ilist, q, p, optimizer_q, temp, epsilon):

    if args.q_objective == "sleep":
        loss = sleep(args, q, p)
    else:
        if args.q_objective == "iw":
            X_repeat = torch.repeat_interleave(X_batch, args.batchsz_iw, dim=0)
            V_q = q.sampleV(X_repeat.shape[0], "full", temp=1, epsilon=0,
                                X=X_repeat, repeat_dags=args.batchsz_iw)
        else:
            V_q = q.sampleV(args.batchsz, args.sampling_dag, temp=temp, epsilon=epsilon,
                                X=X_batch)

        loss = q.objective(p, V_q, ilist)

    optimizer_q.zero_grad()
    loss.backward()
    optimizer_q.step()

    return {"train/loss_q": loss.item()}

def train_model(args, mode, X_batch, ilist, q, p, optimizer_q, optimizer_p, temp, epsilon,
                V_gibbs=None, p_gibbs=None):

    if args.q_objective != "gibbs":
        q.train()
    p.train()

    if mode == "p":
        metrics = train_p(args, X_batch, q, p, optimizer_p, V_gibbs=V_gibbs, p_gibbs=p_gibbs)

    elif mode in "q":
        metrics = train_q(args, X_batch, ilist, q, p, optimizer_q, temp, epsilon)

    return metrics

def eval_model(args, test_set, q, p):

    if args.q_objective != "gibbs":
        q.eval()
    p.eval()

    metrics = {}

    with torch.no_grad():
        samp_data = test_set[torch.randint(len(test_set), (args.batchsz_nll,))].float().to(args.device)

        # NLL
        if args.q_objective != "gibbs":
            nll = get_NLL_importance(q, p, samp_data, args.batchsz_importance_nll).mean().item()
            metrics["eval/NLL"] = nll

    return metrics

def save_model(args, q, p, t):
    torch.save(q.state_dict(), os.path.join(args.save_dir, args.exp_name, f"q_step{t}.pt"))
    torch.save(p.state_dict(), os.path.join(args.save_dir, args.exp_name, f"p_step{t}.pt"))

def main(args):
    makedir(os.path.join(args.save_dir, args.exp_name))
    log_dir = os.path.join(args.path, 'results', args.exp_name, "logs")
    makedir(log_dir)

    # Load dataset
    train_loader, test_loader, test_set = load_mnist(args.batchsz,
                                                     args.batchsz_nll,
                                                     args.n_iters)

    # Load adj
    pyramid = Pyramid(args)
    adj = pyramid.get_adj()
    args.vdim, args.xdim, width, height = pyramid.get_dims()

    # Load PGM
    pgm = PGM(args, adj).to(args.device)

    # Load amortized sampler
    if args.q_objective != "gibbs":
        q = DeltaAI(args, pgm, objective=args.q_objective, model_identity="q").to(args.device)

    # Load model for p
    p = DeltaAI(args, pgm, objective="none", model_identity="p").to(args.device)

    if args.q_objective == "gibbs":
        q = GibbsSampler(p, args.gibbs_steps)
        p_gibbs = 0.5 * torch.ones(args.batchsz, args.vdim - args.xdim, device=args.device)
        V_gibbs = torch.zeros(args.batchsz, args.vdim, device=args.device)
    else:
        p_gibbs = None
        V_gibbs = None

    # OPTIMIZERS
    if args.q_objective != "gibbs":
        param_list_q = [
            {"params": [v for n, v in q.Qnet.named_parameters() if "marginals" in n],
             "lr": args.marg_q_lr},
            {"params": [v for n, v in q.Qnet.named_parameters() if "marginals" not in n],
             "lr": args.q_lr},
        ]

    param_list_p = [
        {"params": [v for n, v in p.Qnet.named_parameters() if "marginals" in n],
         "lr": args.marg_p_lr},
        {"params": [v for n, v in p.Qnet.named_parameters() if "marginals" not in n],
         "lr": args.p_lr}
    ]

    if args.q_objective == "gibbs":
        optimizer_q = None
    else:
        optimizer_q = torch.optim.Adam(param_list_q)

    optimizer_p = torch.optim.Adam(param_list_p)

    p_ilist_F = torch.ones(args.vdim)
    p_ilist_F[:args.xdim] = 0

    t_since_last_eval = 0
    t_since_last_save = 0
    eval_iter = True

    T = lambda t: int(args.n_iters * t)
    decay_steps_gfn = [T(0.4), T(0.7), T(0.9)]

    mode = "p"
    n_iters = {"p": 0, "q": 0}
    max_iters = {"p": args.p_iters, "q": args.q_iters}

    for t, X_batch in enumerate(train_loader):

        if t in decay_steps_gfn:
            # Reduce the learning rates for q and p by 0.5
            if optimizer_q is not None:
                for param_group in optimizer_q.param_groups:
                    param_group['lr'] *= 0.5
            if optimizer_p is not None:
                for param_group in optimizer_p.param_groups:
                    param_group['lr'] *= 0.5

        # ---- SAMPLING DAGs and NODE LIST TO PERTURB ----
        if args.alg == "fixed" or (args.alg == "rand" and t % args.sample_dags_every == 0):
            ilist = get_ilist_and_set_dags(args, pgm, p_ilist_F)


        # ---- TRAINING ----
        X_batch = X_batch[0].to(dtype=torch.float32, device=args.device)
        train_metrics = train_model(args, mode, X_batch, ilist, q, p, optimizer_q, optimizer_p, args.temp, args.epsilon,
                                    V_gibbs=V_gibbs, p_gibbs=p_gibbs)

        if t % args.print_every == 0:
            print(f"t={t}, {', '.join(k + ' ' + str(val) for k, val in train_metrics.items())}")

        if args.q_objective != "gibbs":
            n_iters[mode] += 1

            if n_iters[mode] == max_iters[mode]:
                n_iters[mode] = 0
                if mode == "p":
                    # Evaluate at the end of p training
                    eval_iter = True

                mode = "q" if mode == "p" else "p"

        # ---- EVALUATION ----
        if (eval_iter and t_since_last_eval >= args.print_every) or t == 0:
            eval_metrics = eval_model(args, test_set, q, p)

            print(f"t={t}, {', '.join(k + ' ' + str(val) for k, val in eval_metrics.items())}")

            t_since_last_eval = 0
            if args.q_objective != "gibbs":
                eval_iter = False

            # ---- SAVING ----
            if t_since_last_save >= args.save_every:
                save_model(args, q, p, t)
                t_since_last_save = 0
            else:
                t_since_last_save += 1

        else:
            t_since_last_eval += 1
            t_since_last_save += 1

    # Save the trained model
    torch.save(q.state_dict(), os.path.join(args.save_dir, args.exp_name, f"q_step{args.n_iters}.pt"))
    torch.save(p.state_dict(), os.path.join(args.save_dir, args.exp_name, f"p_step{args.n_iters}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=".")
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--exp_name', type=str, default="test_experiment")
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10000)

    # model
    parser.add_argument('--K', nargs='+', type=int, default=[8, 4, 2])
    parser.add_argument('--stride', nargs='+', type=int, default=[4, 2, 1])
    parser.add_argument('--h_depth', type=int, default=4)
    parser.add_argument('--hdim_mae', type=int, default=512)

    # training
    parser.add_argument('--q_objective', type=str, default="delta",
                        choices=["delta", "iw", "tb", "db", "fldb", "mean_field", "gibbs", "sleep"])
    parser.add_argument('--n_iters', type=int, default=50000)
    parser.add_argument('--batchsz_nll', type=int, default=100)
    parser.add_argument('--batchsz_proc', type=int, default=1000)
    parser.add_argument('--batchsz_importance_nll', type=int, default=100)
    parser.add_argument('--batchsz_iw', type=int, default=7)
    parser.add_argument('--q_lr', type=float, default=1e-3)
    parser.add_argument('--p_lr', type=float, default=1e-3)
    parser.add_argument('--marg_q_lr', type=float, default=1e-1)
    parser.add_argument('--marg_p_lr', type=float, default=1e-1)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--temp', type=float, default=4.0)
    parser.add_argument('--q_iters', type=int, default=100)
    parser.add_argument('--p_iters', type=int, default=100)
    parser.add_argument('--gibbs_steps', type=int, default=1)

    # algorithm
    parser.add_argument('--alg', type=str, default="rand", choices=["fixed", "rand"])
    parser.add_argument('--sampling_dag', type=str, default="partial", choices=["full", "partial"])

    # DAG
    parser.add_argument('--sample_dags_every', type=int, default=10)
    parser.add_argument('--n_nodes', type=int, default=164)
    parser.add_argument('--n_subdags_per_node', type=int, default=1)
    parser.add_argument('--repeat_nodes', type=int, default=7)
    parser.add_argument('--n_fulldags_per_batch', type=int, default=1)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.temp <= 0:
        raise AssertionError("temp must be > 0")

    if len(args.K) < 1:
        raise AssertionError("graph depth must be >= 2")

    args.batchsz = args.repeat_nodes * args.n_nodes
    if args.alg == "rand":
        if args.sampling_dag == "partial":
            args.batchsz *= args.n_subdags_per_node
    else:
        if args.sampling_dag == "partial":
            raise AssertionError("sampling_dag must be full when alg is fixed")

    if args.n_fulldags_per_batch > 1:
        if args.alg == "fixed":
            raise AssertionError
        if args.batchsz % args.n_fulldags_per_batch != 0:
            raise AssertionError("batchsz=n_nodes*repeat_nodes must be divisible by n_fulldags_per_batch")

    if args.q_objective == "gibbs":
        if args.gibbs_steps <= 0:
            raise AssertionError("gibbs_steps must be > 0 when q_objective is gibbs")

    if args.sampling_dag == "partial" and args.q_objective != "delta":
        raise AssertionError("Only delta is compatible with partial inference")

    main(args)

