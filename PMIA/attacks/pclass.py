import numpy as np
import scipy.stats
from utils.shadow_utils import get_shadow_res
from utils.loader import load_dataset
from attacks.util_pmia import pmia_args, likelihood_ratio


def attack(dataset, n_shadows, test_logits):

    all_target_logits, _, n_target = get_shadow_res(dataset, n_shadows, attack_type="pmia", target=True)
    all_shadow_logits, shadow_keep, n_shadow = get_shadow_res(dataset, n_shadows, attack_type="pmia", target=False)

    args = pmia_args(dataset, n_shadows)
    shadow_ds = load_dataset(args, data_type="shadow")
    # Get labels from training dataset
    shadow_labels = np.array([label for _, label in shadow_ds])
    num_classes = len(np.unique(shadow_labels))
    class_in_logits = [[] for _ in range(num_classes)]
    class_in_logits_mean = [[] for _ in range(num_classes)]
    class_in_logits_std = [[] for _ in range(num_classes)]

    for j in range(n_shadow):
        cur_data_in = all_shadow_logits[shadow_keep[:, j], j, :]
        cur_label = shadow_labels[j]
        class_in_logits[cur_label].append(cur_data_in)

    for i in range(num_classes):
        class_in_logits[i] = np.concatenate(class_in_logits[i], axis=0)
        class_in_logits_mean[i] = np.median(class_in_logits[i], axis=0)
        class_in_logits_std[i] = np.std(class_in_logits[i], axis=0)

    target_ds = load_dataset(args, data_type="target")
    target_labels = np.array([label for _, label in target_ds])
    final_score = []
    for j in range(n_target):
        logits_out = all_target_logits[:, j, :]
        cur_label = target_labels[j]
        in_mean = class_in_logits_mean[cur_label]
        in_std = class_in_logits_std[cur_label]

        lira_score = likelihood_ratio(test_logits[j], in_mean, in_std, logits_out)
        final_score.append(lira_score)

    final_score = np.array(final_score)
    np.save(f"{args.dataset}_pclass_score.npy", final_score)
    return final_score
