import pickle
import numpy as np
import argparse
import os
import importlib
from utils.metric import eval_attack
from utils.shadow_utils import get_test

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--attack", default="calibrate", type=str)
parser.add_argument("--n_shadows", default=64, type=int)

args = parser.parse_args()

print(args)
test_logits, test_mask = get_test(args.dataset, args.n_shadows, attack_type=args.attack, target=True)

attack_module = importlib.import_module(f"attacks.{args.attack}")
final_score = attack_module.attack(args.dataset, args.n_shadows, test_logits)

auc, acc, lw0, lw1, lw2, lw3, lw4 = eval_attack(final_score, test_mask.astype(bool))

print(
    f"Final {args.attack.upper()} attack \nAUC = {auc}\nACC = {acc}\nTPR@10%FPR: {lw0}\nTPR@1%FPR: {lw1}\nTPR@0.1%FPR: {lw2}\nTPR@0.01%FPR: {lw3} \nTPR@0.001%FPR: {lw4}",
)
