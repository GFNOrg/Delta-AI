from torch.optim import Adam
from algorithms.VAE import VAE
import torch
import os
import argparse
from copy import deepcopy
from data import load_mnist
from utils.misc import makedir
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10000)

    # data & model
    parser.add_argument('--hdim', type=int, default=164)

    # training
    parser.add_argument('--n_iters', type=int, default=1000000)
    parser.add_argument('--batchsz', type=int, default=1000)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # evaluation
    parser.add_argument('--batchsz_nll', type=int, default=100)
    parser.add_argument('--batchsz_importance_nll', type=int, default=100)

    args = parser.parse_args()

    device = torch.device("cpu") if args.gpu < 0 else torch.device(f"cuda:{args.gpu}")

    makedir(args.save_dir)
    makedir(os.path.join(args.save_dir, args.exp_name))

    # Load dataset
    train_loader, test_loader, test_set = load_mnist(args.batchsz,
                                                     args.batchsz_nll,
                                                     args.n_iters)

    # Visualize ground truth samples
    samp_data = test_set[torch.randint(len(test_set), (10,))].to(device)

    # Initialize VAE model
    model = VAE(in_dim=28*28, latent_dim=args.hdim).to(device)

    # Define optimizer with weight decay
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    T = lambda t: int(args.n_iters * t)
    decay_steps = [T(0.2), T(0.4), T(0.6), T(0.8), T(0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, decay_steps, gamma=0.1)

    # Training loop
    model.train()
    originaltime = time.time()
    for t, X_batch in enumerate(train_loader):

        X_batch = X_batch[0].to(device).float()

        # Compute reconstruction loss and KL divergence
        rec_data, mu, logvar = model((X_batch + 1.)/2.)
        loss = model.loss_function(rec_data, X_batch, mu, logvar)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training progress
        scheduler.step()

        if t % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                samp_data = test_set[torch.randint(len(test_set), (args.batchsz_nll,))].float().to(args.device)
                nll = model.NLLimportance(samp_data, args.batchsz_importance_nll).mean()
                print(f"Step {t} | NLL: {nll:.2f} | Loss: {loss.item():.2f} | Time: {time.time() - originaltime:.2f}")

            model.train()

        if t % args.save_every == 0:
            torch.save(deepcopy(model), os.path.join(args.save_dir, args.exp_name, f"vae_step{t}.pt"))

    # Final save
    torch.save(deepcopy(model), os.path.join(args.save_dir, args.exp_name, f"vae_step{t}.pt"))
