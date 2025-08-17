import numpy as np
import scipy.stats
from utils.shadow_utils import get_all_shadow_models


def attack(dataset, n_shadows, cur_round, test_logits):
    all_logits, all_keep, n_samples = get_all_shadow_models(dataset, n_shadows, cur_round)
    final_score = []

    for j in range(n_samples):
        logits_in = all_logits[all_keep[:, j], j, :]
        logits_out = all_logits[~all_keep[:, j], j, :]
        lira_score = likelihood_ratio(test_logits[j], logits_in, logits_out)
        final_score.append(lira_score)

    final_score = np.array(final_score)
    return final_score


def likelihood_ratio(target_margins, in_margin, out_margin):
    """
    likelihood ratio attack
    use mean and std of IN/OUT to calculate the log pdf of target
    """
    _in_mean = np.median(in_margin, 0)  # there are many queries
    _in_std = np.std(in_margin, 0)
    _out_mean = np.median(out_margin, 0)
    _out_std = np.std(out_margin, 0)

    # if abs(in_margin.shape[0] - out_margin.shape[0]) > 10:
    #     print(f"IN model size: {in_margin.shape}, OUT model size: {out_margin.shape}")

    logp_in = scipy.stats.norm.logpdf(target_margins, _in_mean, _in_std + 1e-30)
    logp_out = scipy.stats.norm.logpdf(target_margins, _out_mean, _out_std + 1e-30)

    score = logp_in - logp_out
    return score.mean()
