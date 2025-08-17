import pickle
import numpy as np
import argparse
import os
import importlib
from utils.shadow_utils import get_reference, get_test, get_anchors, get_all_shadow_models
from utils.cmia_utils import determine_negative_threshold, determine_positive_threshold

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--attack", default="lira", type=str)
parser.add_argument("--cur_round", "-cr", default=1, type=int)
parser.add_argument("--n_shadows", default=256, type=int)
parser.add_argument("--n_reference", default=1, type=int)
parser.add_argument("--positive_tolerance", default=1, type=int)
parser.add_argument("--negative_tolerance", default=1, type=int)
parser.add_argument("--max_anchor_ratio", default=0.05, type=float)
parser.add_argument("--min_anchor_size", default=20, type=int)

args = parser.parse_args()

print(args)


attack_module = importlib.import_module(f"attacks.{args.attack}")

shadow_dir = f"{args.dataset}/r{args.cur_round}_exp/"
anchor_mask_dict = get_anchors(args.dataset, args.cur_round)
ref_logits, ref_masks = get_reference(args.dataset, args.n_shadows, args.cur_round, args.n_reference, attack_type=args.attack)
test_logits, _ = get_test(args.dataset, args.n_shadows, args.n_reference, attack_type=args.attack)
n_samples = len(test_logits)

#######################################
# use reference models to determine thresholds
print(f"compute the MIA score for reference models ...")
all_neg_thresholds = []
all_pos_thresholds = []
n_refs = ref_logits.shape[0]
for ref_idx in range(ref_logits.shape[0]):
    cur_scores = []
    cur_labels = []

    # step1: compute the membership score using current reference model
    mia_score = attack_module.attack(args.dataset, args.n_shadows, args.cur_round, ref_logits[ref_idx])
    for i in range(n_samples):
        if i in anchor_mask_dict:
            continue
        cur_scores.append(mia_score[i])
        cur_labels.append(ref_masks[ref_idx, i])

    # step2: determine thresholds using membership from reference model
    cur_scores = np.array(cur_scores)
    cur_labels = np.array(cur_labels)

    neg_threshold = determine_negative_threshold(cur_scores, cur_labels, args.negative_tolerance)
    pos_threshold = determine_positive_threshold(cur_scores, cur_labels, args.positive_tolerance)

    if neg_threshold is not None:
        all_neg_thresholds.append(neg_threshold)
    if pos_threshold is not None:
        all_pos_thresholds.append(pos_threshold)

# Average the thresholds
neg_threshold = np.mean(all_neg_thresholds) if len(all_neg_thresholds) > 0 else None
pos_threshold = np.mean(all_pos_thresholds) if len(all_pos_thresholds) > 0 else None
print(f"all_neg_thresholds: {all_neg_thresholds}")
print(f"all_pos_thresholds: {all_pos_thresholds}")
print(f"neg_threshold: {neg_threshold}, pos_threshold: {pos_threshold}")
#######################################


# use the threshold on target model
# step1: compute the LiRA score using target model
mia_score = attack_module.attack(args.dataset, args.n_shadows, args.cur_round, test_logits)
test_indices, test_scores = [], []
for i in range(n_samples):
    if i in anchor_mask_dict:
        continue
    test_indices.append(i)
    test_scores.append(mia_score[i])

test_indices = np.array(test_indices)
test_scores = np.array(test_scores)

# step2: determine the anchor samples
anchored_dict = {}
max_anchor_size = int(len(test_indices) * args.max_anchor_ratio)

# First handle negative anchors
if neg_threshold is not None:
    print(f"determine negative anchors ...")
    nonmember_mask = test_scores <= neg_threshold
    nonmember_indices = test_indices[nonmember_mask]

    # truncate the number of negative anchors
    if len(nonmember_indices) > max_anchor_size:
        # Sort by score (ascending) and take only max_anchor_size samples
        sorted_indices = nonmember_indices[np.argsort(test_scores[nonmember_mask])[:max_anchor_size]]
        nonmember_indices = sorted_indices

    if len(nonmember_indices) < args.min_anchor_size:
        print(f"number of negative anchors is {len(nonmember_indices)}, discard them")
        neg_threshold = None
    else:
        # update negative anchors
        for i in nonmember_indices:
            anchored_dict[i] = 0  # Non-member

# Only consider positive anchors if:
# 1. no negative threshold is found
# 2. positive threshold is found
if neg_threshold is None and pos_threshold is not None:
    print(f"determine positive anchors ...")

    member_mask = test_scores >= pos_threshold
    member_indices = test_indices[member_mask]

    # if len(member_indices) < args.min_anchor_size:
    if len(member_indices) < 5:  # since the threshold is more strict, so we set a smaller min_anchor_size
        print(f"number of positive anchors is {len(member_indices)}, discard them")
        pos_threshold = None
    else:
        # Add positive anchors to dictionary
        for i in member_indices:
            anchored_dict[i] = 1  # Member
else:
    print(f"no positive anchors are determined")
    pos_threshold = None
#######################################


if len(anchored_dict) == 0:
    print("Could not find valid thresholds meeting the requirements. Stopping the cascade.")
    exit(0)

print(f"Automatically determined thresholds for round {args.cur_round}: pos={pos_threshold}, neg={neg_threshold}")

# update the mask
for i, mask in anchored_dict.items():
    anchor_mask_dict[i] = mask

count_mask_in = sum(value == 1 for value in anchored_dict.values())
print(f"# IN/OUT anchor: {count_mask_in}/{len(anchored_dict) - count_mask_in}")


# save the new mask and score
pickle.dump(anchor_mask_dict, open(f"{args.dataset}/anchor_mask_r{args.cur_round}.pkl", "wb"))
print(f"saved new mask to {args.dataset}/anchor_mask_r{args.cur_round}.pkl")
