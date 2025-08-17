import pickle
import numpy as np
import argparse
import os
import importlib
from utils.metric import eval_attack, compute_metric
from utils.loader import load_mask
from utils.shadow_utils import get_test

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--attack", default="lira", type=str)
parser.add_argument("--cur_round", "-cr", default=1, type=int)
parser.add_argument("--n_shadows", default=256, type=int)
parser.add_argument("--n_reference", default=5, type=int)

args = parser.parse_args()

print(args)

test_logits, test_mask = get_test(args.dataset, args.n_shadows, args.n_reference, attack_type=args.attack)

attack_module = importlib.import_module(f"attacks.{args.attack}")
final_score = attack_module.attack(args.dataset, args.n_shadows, args.cur_round, test_logits)

auc, acc, lw0, lw1, lw2, lw3, lw4 = eval_attack(final_score, test_mask.astype(bool))

fpr, tpr, auc, acc = compute_metric(final_score, test_mask.astype(bool))

auc, acc, lw0, lw1, lw2, lw3, lw4 = eval_attack(final_score, test_mask.astype(bool))

plot_dict = {
    "fpr": fpr,
    "tpr": tpr,
}

if args.cur_round == 1:
    filename = f"{args.dataset}_{args.attack}_plot_dict.pkl"
else:
    filename = f"{args.dataset}_{args.attack}_cmia_plot_dict.pkl"
with open(filename, "wb") as f:
    pickle.dump(plot_dict, f)


print(
    f"Final {args.attack.upper()} attack (round={args.cur_round}) \nAUC = {auc}\nACC = {acc}\nTPR@10%FPR: {lw0}\nTPR@1%FPR: {lw1}\nTPR@0.1%FPR: {lw2}\nTPR@0.01%FPR: {lw3} \nTPR@0.001%FPR: {lw4}",
)
