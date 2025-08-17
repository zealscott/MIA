import argparse
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from utils.loader import load_labels, load_dataset, load_model, load_mask
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metric import get_acc

parser = argparse.ArgumentParser()
# model parameters
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--model_type", default="resnet", type=str)
# mia parameters
parser.add_argument("--n_shadows", default=256, type=int)
parser.add_argument("--n_queries", default=18, type=int)
parser.add_argument("--shadow_id", default=0, type=int)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--data_dir", default="/path/to/datasets", type=str)
# cascading attack parameters
parser.add_argument("--n_reference", default=1, type=int)
parser.add_argument("--keep_path", default="r1_exp/", type=str)  # the keep mask for the previous round
parser.add_argument("--savedir", default="r1_exp/", type=str)
args = parser.parse_args()
print(args)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


def train():
    print(f"Training {args.shadow_id} shadow model...")
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    pl.seed_everything(seed)

    train_ds, test_ds = load_dataset(args)

    # Compute the IN / OUT subset:
    # If we run each experiment independently then even after a lot of trials
    # there will still probably be some examples that were always included
    # or always excluded. So instead, with experiment IDs, we guarantee that
    # after `args.n_shadows` are done, each example is seen exactly half
    # of the time in train, and half of the time not in train.

    size = len(train_ds)
    np.random.seed(2025)

    # First handle shadow models to ensure half IN/OUT distribution
    keep_shadows = np.random.uniform(0, 1, size=(args.n_shadows, size))
    order_shadows = keep_shadows.argsort(0)
    keep_shadows = order_shadows < int(args.pkeep * args.n_shadows)

    # Separately handle reference and target models
    total_extra = args.n_reference
    if args.shadow_id < args.n_shadows:
        # For shadow models, use the balanced distribution
        keep = np.array(keep_shadows[args.shadow_id], dtype=bool)
    else:
        # For reference/target models, sample all at once with same probability
        extra_id = args.shadow_id - args.n_shadows
        keep_extra = np.random.uniform(0, 1, size=(total_extra, size)) < args.pkeep
        keep = np.array(keep_extra[extra_id], dtype=bool)

    ####################################################
    # here we force the previous determined keep mask
    print(f"before anchoring, keep.shape: {keep.shape}, keep.sum(): {keep.sum()}")
    force_in, force_out = load_mask(args.keep_path)
    keep[force_in] = True
    keep[force_out] = False
    print(f"after anchoring, keep.shape: {keep.shape}, keep.sum(): {keep.sum()}")
    ####################################################

    keep = keep.nonzero()[0]

    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(train_ds, keep)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    m = load_model(args).to(device)

    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Train
    for i in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            if x.size(0) == 1:  # Skip batch if batch size is 1
                continue
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(m(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

    print(f"test accuracy: {get_acc(m, test_dl, device):.4f}")

    savedir = os.path.join(args.savedir, str(args.shadow_id))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")
    print(f"saved model for {args.shadow_id} shadow model")


@torch.no_grad()
def inference():
    print(f"inferring {args.shadow_id} shadow model...")
    train_ds, _ = load_dataset(args)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)

    m = load_model(args)
    m.load_state_dict(torch.load(os.path.join(args.savedir, str(args.shadow_id), "model.pt")))
    m.to(device)
    m.eval()

    logits_n = []
    for i in range(args.n_queries):
        logits = []
        for x, _ in tqdm(train_dl):
            x = x.to(device)
            outputs = m(x)
            logits.append(outputs.cpu().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)
    print(f"shape of logits: {logits_n.shape}")  # [n_samples, n_queries, n_classes]

    # the following converts it to a scored prediction
    # in lira the scaling_logit = logp_in - logp_out
    # be exceptionally careful
    # numerically stable everything, as described in the paper
    predictions = logits_n - np.max(logits_n, axis=-1, keepdims=True)
    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

    labels = load_labels(args)

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

    print("training acc", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    print(f"final logit scaling shape: {logit.shape}")  # [n_examples, n_queries]
    np.save(os.path.join(args.savedir, str(args.shadow_id), "scaled_logits.npy"), logit)
    print(f"saved scaled logits for {args.shadow_id} shadow model")


def rmia_inference(temperature: float = 2.0):
    print(f"inferring {args.shadow_id} shadow model...")
    train_ds, test_ds = load_dataset(args)
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)
    m = load_model(args)
    m.load_state_dict(torch.load(os.path.join(args.savedir, str(args.shadow_id), "model.pt")))
    m.to(device)
    m.eval()

    train_softmax_scores_n = []
    for i in range(args.n_queries):
        train_softmax_scores = []
        for x, _ in tqdm(train_dl):
            x = x.to(device)
            outputs = m(x)
            outputs = torch.softmax(outputs / temperature, dim=1)
            train_softmax_scores.append(outputs.detach().cpu().numpy())
        train_softmax_scores_n.append(np.concatenate(train_softmax_scores))
    train_softmax_scores_n = np.stack(train_softmax_scores_n, axis=1)
    np.save(os.path.join(args.savedir, str(args.shadow_id), "train_softmax_scores.npy"), train_softmax_scores_n)
    print(f"saved train softmax scores for {args.shadow_id} shadow model")

    test_softmax_scores_n = []
    for i in range(args.n_queries):
        test_softmax_scores = []
        for x, _ in tqdm(test_dl):
            x = x.to(device)
            outputs = m(x)
            outputs = torch.softmax(outputs / temperature, dim=1)
            test_softmax_scores.append(outputs.detach().cpu().numpy())
        test_softmax_scores_n.append(np.concatenate(test_softmax_scores, axis=0))
    test_softmax_scores_n = np.stack(test_softmax_scores_n, axis=1)
    np.save(os.path.join(args.savedir, str(args.shadow_id), "test_softmax_scores.npy"), test_softmax_scores_n)
    print(f"saved test softmax scores for {args.shadow_id} shadow model")



def loss_inference():
    print(f"inferring {args.shadow_id} shadow model...")
    train_ds, _ = load_dataset(args)
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=4)

    m = load_model(args)
    m.load_state_dict(torch.load(os.path.join(args.savedir, str(args.shadow_id), "model.pt")))
    m.to(device)
    m.eval()

    train_losses = []
    for x, y in tqdm(train_dl):
        x = x.to(device)
        y = y.to(device)
        outputs = m(x)
        # Use reduction='none' to get per-sample losses
        loss = F.cross_entropy(outputs, y, reduction='none')
        train_losses.append(loss.detach().cpu().numpy())
    
    # Concatenate all batch losses into a single array of per-sample losses
    train_losses = np.concatenate(train_losses)
    np.save(os.path.join(args.savedir, str(args.shadow_id), "train_losses.npy"), train_losses)
    print(f"saved train losses for {args.shadow_id} shadow model")
    print(f"shape of train losses: {train_losses.shape}")  # Should be [n_samples]

if __name__ == "__main__":
    if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "model.pt")):
        print(f"already trained shadow model {args.shadow_id} one {args.savedir}, skip training")
    else:
        train()

    if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "scaled_logits.npy")):
        print(f"already get scaled logits for shadow model {args.shadow_id} one {args.savedir}, skip inference")
    else:
        inference()

    if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "train_softmax_scores.npy")):
        print(f"already get train softmax scores for shadow model {args.shadow_id} one {args.savedir}, skip inference")
    else:
        rmia_inference()

    if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "train_losses.npy")):
        print(f"already get train losses for shadow model {args.shadow_id} one {args.savedir}, skip inference")
    else:
        loss_inference()
