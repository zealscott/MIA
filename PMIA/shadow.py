import argparse
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from utils.loader import load_labels, load_dataset, load_model
from utils.metric import get_acc
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
# model parameters
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--model_type", default="resnet", type=str)
# mia parameters
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--n_shadows", default=256, type=int)
parser.add_argument("--n_queries", default=18, type=int)
parser.add_argument("--shadow_id", default=0, type=int)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--data_dir", default="/path/to/datasets", type=str)
parser.add_argument("--savedir", default="r1_exp/", type=str)
args = parser.parse_args()
print(args)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


def train():
    print(f"Training {args.shadow_id} shadow model...")
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    pl.seed_everything(seed)

    # the target model is trained disjointly
    if args.shadow_id == args.n_shadows:
        data_ds = load_dataset(args, data_type="target")
    else:
        data_ds = load_dataset(args, data_type="shadow")

    # Compute the IN / OUT subset:
    # If we run each experiment independently then even after a lot of trials
    # there will still probably be some examples that were always included
    # or always excluded. So instead, with experiment IDs, we guarantee that
    # after `args.n_shadows` are done, each example is seen exactly half
    # of the time in train, and half of the time not in train.

    size = len(data_ds)
    np.random.seed(2025)

    if args.shadow_id == args.n_shadows:
        keep_shadows = np.random.uniform(0, 1, size=size)
        keep = np.array(keep_shadows < args.pkeep, dtype=bool)
    else:
        keep_shadows = np.random.uniform(0, 1, size=(args.n_shadows, size))
        order_shadows = keep_shadows.argsort(0)
        keep_shadows = order_shadows < int(args.pkeep * args.n_shadows)
        keep = np.array(keep_shadows[args.shadow_id], dtype=bool)

    keep = keep.nonzero()[0]

    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(data_ds, keep)
    test_ds = torch.utils.data.Subset(data_ds, ~keep)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    m = load_model(args).to(device)

    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_acc = 0
    savedir = os.path.join(args.savedir, str(args.shadow_id))
    os.makedirs(savedir, exist_ok=True)
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

        # test on test_ds
        m.eval()
        acc = get_acc(m, test_dl, device)
        if acc > best_acc:
            best_acc = acc
            print(f"epoch {i} best acc: {best_acc:.4f}")
            torch.save(m.state_dict(), savedir + "/model.pt")

    print(f"test accuracy: {get_acc(m, test_dl, device):.4f}")
    np.save(savedir + "/keep.npy", keep_bool)
    print(f"saved model for {args.shadow_id} shadow model")


@torch.no_grad()
def inference(data_type="target"):
    """
    the difference between hard_offline and online is that
    the shadow model should be inferred on the target train dataset
    """
    print(f"inferring {args.shadow_id} shadow model...")
    # the shadow model should be inferred on the target train dataset
    data_ds = load_dataset(args, data_type=data_type)
    data_dl = DataLoader(data_ds, batch_size=512, shuffle=False, num_workers=4)

    m = load_model(args)
    m.load_state_dict(torch.load(os.path.join(args.savedir, str(args.shadow_id), "model.pt")))
    m.to(device)
    m.eval()

    logits_n = []
    for i in range(args.n_queries):
        logits = []
        for x, _ in tqdm(data_dl):
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

    labels = load_labels(data_ds)

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

    print(f"acc on {data_type} dataset", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    print(f"final logit scaling shape: {logit.shape}")  # [n_examples, n_queries]
    if data_type == "target":
        np.save(os.path.join(args.savedir, str(args.shadow_id), "scaled_logits_target.npy"), logit)
    else:
        np.save(os.path.join(args.savedir, str(args.shadow_id), "scaled_logits_shadow.npy"), logit)
    print(f"saved scaled logits for {args.shadow_id} shadow model")


def rmia_inference(data_type="target", temperature=2.0):
    print(f"inferring softmax scores for {args.shadow_id} shadow model...")
    data_ds = load_dataset(args, data_type=data_type)
    data_dl = DataLoader(data_ds, batch_size=512, shuffle=False, num_workers=4)

    m = load_model(args)
    m.load_state_dict(torch.load(os.path.join(args.savedir, str(args.shadow_id), "model.pt")))
    m.to(device)
    m.eval()

    softmax_scores_n = []
    for i in range(args.n_queries):
        softmax_scores = []
        for x, _ in tqdm(data_dl):
            x = x.to(device)
            outputs = m(x)
            outputs = torch.softmax(outputs / temperature, dim=1)
            softmax_scores.append(outputs.detach().cpu().numpy())
        softmax_scores_n.append(np.concatenate(softmax_scores))
    softmax_scores_n = np.stack(softmax_scores_n, axis=1)
    if data_type == "target":
        np.save(os.path.join(args.savedir, str(args.shadow_id), "softmax_scores_target.npy"), softmax_scores_n)
    else:
        np.save(os.path.join(args.savedir, str(args.shadow_id), "softmax_scores_shadow.npy"), softmax_scores_n)
    print(f"saved softmax scores for {args.shadow_id} shadow model")


def loss_inference(data_type="target"):
    data_ds = load_dataset(args, data_type=data_type, augment=False)
    data_dl = DataLoader(data_ds, batch_size=512, shuffle=False, num_workers=4)

    m = load_model(args)
    m.load_state_dict(torch.load(os.path.join(args.savedir, str(args.shadow_id), "model.pt")))
    m.to(device)
    m.eval()

    train_losses = []
    for x, y in tqdm(data_dl):
        x = x.to(device)
        y = y.to(device)
        outputs = m(x)
        # Use reduction='none' to get per-sample losses
        loss = F.cross_entropy(outputs, y, reduction="none")
        train_losses.append(loss.detach().cpu().numpy())

    # Concatenate all batch losses into a single array of per-sample losses
    train_losses = np.concatenate(train_losses)
    if data_type == "target":
        np.save(os.path.join(args.savedir, str(args.shadow_id), "losses_target.npy"), train_losses)
    else:
        np.save(os.path.join(args.savedir, str(args.shadow_id), "losses_shadow.npy"), train_losses)
    print(f"saved losses for {args.shadow_id} shadow model")


if __name__ == "__main__":
    if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "keep.npy")):
        print(f"already trained shadow model {args.shadow_id} one {args.savedir}, skip training")
    else:
        train()

    if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "scaled_logits_shadow.npy")):
        print(f"already inferred shadow model {args.shadow_id} one {args.savedir}, skip inference")
    else:
        inference(data_type="shadow")

    if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "scaled_logits_target.npy")):
        print(f"already inferred shadow model {args.shadow_id} one {args.savedir}, skip inference")
    else:
        inference(data_type="target")

    # if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "softmax_scores_shadow.npy")):
    #     print(f"already inferred shadow model {args.shadow_id} one {args.savedir}, skip inference")
    # else:
    #     rmia_inference(data_type="shadow")

    # if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "softmax_scores_target.npy")):
    #     print(f"already inferred shadow model {args.shadow_id} one {args.savedir}, skip inference")
    # else:
    #     rmia_inference(data_type="target")

    # if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "losses_shadow.npy")):
    #     print(f"already inferred shadow model {args.shadow_id} one {args.savedir}, skip inference")
    # else:
    #     loss_inference(data_type="shadow")

    # if os.path.exists(os.path.join(args.savedir, str(args.shadow_id), "losses_target.npy")):
    #     print(f"already inferred shadow model {args.shadow_id} one {args.savedir}, skip inference")
    # else:
    #     loss_inference(data_type="target")
