"""
This file is the script to run the CMIA framework (attack part).
"""

import os
import argparse
from utils.loader import load_config


parser = argparse.ArgumentParser()

######### env configuration ########
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--attack", default="calibrate", type=str)
args = parser.parse_args()
load_config(args)
print(args)


command = "python attack.py"

command += f" --dataset {args.dataset} --n_shadows {args.n_shadows} --attack {args.attack}"

os.system(command)
