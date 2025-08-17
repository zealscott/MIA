import numpy as np
import scipy.stats
from utils.shadow_utils import get_shadow_res
from utils.loader import load_dataset
from attacks.util_pmia import pmia_args, likelihood_ratio
from attacks.util_search import find_similar_images_faiss, find_similar_images_clip


def attack(dataset, n_shadows, test_logits):
    topk = 1

    all_target_logits, _, n_target = get_shadow_res(dataset, n_shadows, attack_type="pmia", target=True)
    all_shadow_logits, shadow_keep, n_shadow = get_shadow_res(dataset, n_shadows, attack_type="pmia", target=False)

    args = pmia_args(dataset, n_shadows)
    shadow_logits_in = []
    for j in range(n_shadow):
        cur_data_in = all_shadow_logits[shadow_keep[:, j], j, :]
        shadow_logits_in.append(cur_data_in)

    shadow_ds = load_dataset(args, data_type="shadow", augment=False)
    target_ds = load_dataset(args, data_type="target", augment=False)

    # similar_indices, _, _, _= find_similar_images_faiss(target_ds, shadow_ds, topk=10)
    similar_indices, _, _, _ = find_similar_images_clip(target_ds, shadow_ds, topk=topk)
    final_score = []
    import time

    start_time = time.time()
    # score shape: [n_shadow_models, n_samples, n_queries]
    for j in range(n_target):
        similar_ids = similar_indices[j]
        similar_in_logits = []
        for id in similar_ids:
            similar_in_logits.append(shadow_logits_in[id])
        similar_in_logits = np.array(similar_in_logits)
        in_mean = np.median(similar_in_logits, axis=(0, 1))
        in_std = np.std(similar_in_logits, axis=(0, 1))
        logits_out = all_target_logits[:, j, :]

        lira_score = likelihood_ratio(test_logits[j], in_mean, in_std, logits_out)
        final_score.append(lira_score)

    final_score = np.array(final_score)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    np.save(f"{args.dataset}_pinstance_score_k_{topk}.npy", final_score)
    return final_score
