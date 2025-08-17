import numpy as np
import scipy.stats
from utils.shadow_utils import get_shadow_res
from attacks.util_pmia import likelihood_ratio


def attack(dataset, n_shadows, test_logits):
    all_target_logits, all_keep, n_target = get_shadow_res(dataset, n_shadows, attack_type="pmia", target=True)
    all_shadow_logits, all_keep, n_shadow = get_shadow_res(dataset, n_shadows, attack_type="pmia", target=False)
    data_in = []
    for j in range(n_shadow):
        per_data_in = all_shadow_logits[all_keep[:, j], j, :]
        data_in.append(per_data_in)

    all_data_in = np.concatenate(data_in, axis=0)
    in_mean = np.median(all_data_in, axis=0)
    in_std = np.std(all_data_in, axis=0)

    final_score = []
    for j in range(n_target):
        logits_out = all_target_logits[:, j, :]
        score = likelihood_ratio(test_logits[j], in_mean, in_std, logits_out)
        final_score.append(score)

    final_score = np.array(final_score)
    np.save(f"{dataset}_pglobal_score.npy", final_score)
    return final_score
