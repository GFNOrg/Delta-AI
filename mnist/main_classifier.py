import torch
import os
import argparse

from copy import deepcopy
from algorithms.CNN import CNNClassifier
from data import load_mnist
from utils.misc import Logger, makedir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10000)

    # training
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--batchsz', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    device = torch.device("cpu") if args.gpu < 0 else torch.device(f"cuda:{args.gpu}")

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

    # Load dataset
    train_loader, test_loader, test_set = load_mnist(batch_size_train=args.batchsz,
                                                     batch_size_test=args.batchsz,
                                                     n_iters=args.n_iters)

    # Initialize CNN
    model = CNNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Scheduler
    T = lambda t: int(args.n_iters * t)
    decay_steps = [T(0.4), T(0.7), T(0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, decay_steps, gamma=0.5)

    path_txt = os.path.join(args.save_dir, args.exp_name, "nll.txt")
    if os.path.isfile(path_txt):
        os.remove(path_txt)

    logger.start()
    model.train()
    for t, data in enumerate(train_loader):

        # ---- TRAINING ----
        X_batch = data[0].to(device).float().view(-1, 1, 28, 28)
        target = data[1].to(device)

        logits = model(X_batch)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ---- EVALUATION ----
        if t % args.print_every == 0:
            model.eval()

            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    target = target.to(device)
                    logits = model(data.to(device).float().view(-1, 1, 28, 28))
                    test_loss += criterion(logits, target).item()
                    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    correct += (pred == target).sum()

            test_loss /= len(test_loader.dataset)
            logger.meter("test", "nll", test_loss)
            acc = correct / len(test_loader.dataset)
            logger.meter("test", "accuracy", acc)

            model.train()

        # ---- SAVING ----
        if t % args.save_every == 0:
            torch.save(deepcopy(model), os.path.join(args.save_dir, args.exp_name, f"model_step{t}.pt"))

        logger.step()

    logger.finish()

    # Save the trained model
    torch.save(deepcopy(model), os.path.join(args.save_dir, args.exp_name, f"model_step{args.n_iters}.pt"))