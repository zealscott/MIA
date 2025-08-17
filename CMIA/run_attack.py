"""
This file is the script to run the CMIA framework (attack part).
"""

import os
import argparse
from utils.loader import load_config


parser = argparse.ArgumentParser()

######### env configuration ########
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--cur_round", "-cr", default=1, type=int)
parser.add_argument("--attack", default="lira", type=str)
args = parser.parse_args()
load_config(args)
print(args)


command = "python attack.py"

command += f" --dataset {args.dataset} --cur_round {args.cur_round} --n_shadows {args.n_shadows} --n_reference {args.n_reference} --attack {args.attack}"

os.system(command)
