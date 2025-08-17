import os
import numpy as np
import pickle
from typing import Tuple, Dict, List, Union


def mia_output(attack_type):
    if attack_type == "rmia":
        return "train_softmax_scores.npy"
    else:
        return "scaled_logits.npy"


def get_all_shadow_models(
    dataset: str, n_shadows: int, cur_round: int, attack_type: str = "lira"
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Aggregate all shadow models from previous rounds.

    Args:
        dataset (str): Name of the dataset (e.g., 'cifar10')
        n_shadows (int): Number of shadow models per round
        cur_round (int): Current round number
        attack_type (str): Type of attack to determine which scores to load

    Returns:
        - all_logits (np.ndarray): Array of shape (n_rounds * n_shadows, n_samples, n_augmentations)
            containing logits from all shadow models
        - all_keep (np.ndarray): Array of shape (n_rounds * n_shadows, n_samples) containing
            membership masks for all shadow models
        - n_samples (int): Number of samples in the dataset
    """
    # load all shadow models
    all_logits = []
    all_keep = []
    for round in range(1, cur_round + 1):
        shadow_dir = f"{dataset}/r{round}_exp/"
        cur_logits, cur_keep, n_samples = get_shadow_res(shadow_dir, n_shadows, attack_type)
        all_logits.append(cur_logits)
        all_keep.append(cur_keep)

    all_logits = np.concatenate(all_logits, axis=0)
    all_keep = np.concatenate(all_keep, axis=0)
    # print(f"shape of all_logits: {all_logits.shape}, shape of all_keep: {all_keep.shape}")

    return all_logits, all_keep, n_samples


def get_test(
    dataset: str, n_shadows: int, n_reference: int, attack_type: str = "lira"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the test models which are the last models in the first round.

    Args:
        dataset (str): Name of the dataset (e.g., 'cifar10')
        n_shadows (int): Number of shadow models
        n_reference (int): Number of reference models

    Returns:
        - test_logits (np.ndarray): Array of shape (n_samples, n_augmentations) containing logits
            from the test model
        - test_masks (np.ndarray): Array of shape (n_samples, ) containing membership mask
            for the test model
    """
    print(f"load {dataset}/r1_exp/{n_shadows + n_reference} as test model ...")
    res_file = mia_output(attack_type)
    test_logits = np.load(f"{dataset}/r1_exp/{n_shadows + n_reference}/{res_file}")
    test_masks = np.load(f"{dataset}/r1_exp/{n_shadows + n_reference}/keep.npy")

    # print(f"shape of test_logits: {test_logits.shape}, shape of test_masks: {test_masks.shape}")

    return test_logits, test_masks


def get_reference(
    dataset: str, n_shadows: int, cur_round: int, n_reference: int, attack_type: str = "lira"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the reference models which are the last n_reference models in each round.

    Args:
        dataset (str): Name of the dataset (e.g., 'cifar10')
        n_shadows (int): Number of shadow models
        cur_round (int): Current round number
        n_reference (int): Number of reference models

    Returns:
        - ref_logits (np.ndarray): Array of shape (n_reference, n_samples, n_augmentations) containing
            logits from reference models
        - ref_masks (np.ndarray): Array of shape (n_reference, n_samples) containing membership
            masks for reference models
    """
    shadow_dir = f"{dataset}/r{cur_round}_exp/"
    ref_logits = []
    ref_masks = []

    start_idx = n_shadows
    for i in range(n_reference):
        model_idx = start_idx + i
        res_file = mia_output(attack_type)
        ref_logits.append(np.load(os.path.join(shadow_dir, f"{model_idx}", res_file)))
        ref_masks.append(np.load(os.path.join(shadow_dir, f"{model_idx}", "keep.npy")))

    ref_logits = np.array(ref_logits)
    ref_masks = np.array(ref_masks)
    # print(f"shape of ref_logits: {ref_logits.shape}, shape of ref_masks: {ref_masks.shape}")

    return ref_logits, ref_masks


def get_shadow_res(shadow_dir: str, n_shadow: int, attack_type: str = "lira") -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load the keep mask and scores of the shadow models.

    Args:
        shadow_dir (str): Directory path containing shadow model results
        n_shadow (int): Number of shadow models
        attack_type (str): Type of attack to determine which scores to load

    Returns:
        - shadow_logits (np.ndarray): Array of shape (n_shadow, n_samples, n_augmentations) containing
            logits from shadow models
        - shadow_masks (np.ndarray): Array of shape (n_shadow, n_samples) containing membership
            masks for shadow models
        - n_samples (int): Number of samples in the dataset
    """
    shadow_logits = []
    shadow_masks = []

    for model_idx in range(n_shadow):
        model_dir = os.path.join(shadow_dir, str(model_idx))
        shadow_masks.append(np.load(os.path.join(model_dir, "keep.npy")))

        res_file = mia_output(attack_type)
        shadow_logits.append(np.load(os.path.join(model_dir, res_file)))

    shadow_logits = np.array(shadow_logits)
    shadow_masks = np.array(shadow_masks)

    n_samples = shadow_logits.shape[1]

    return shadow_logits, shadow_masks, n_samples


def get_anchors(dataset: str, cur_round: int) -> Dict[int, int]:
    """
    Get the anchors (membership) from previous rounds.

    Args:
        dataset (str): Name of the dataset (e.g., 'cifar10')
        cur_round (int): Current round number

    Returns:
        Dict[int, int]: Dictionary mapping sample indices to their membership status:
            - 1: Member (in training set)
            - 0: Non-member (not in training set)
            - Empty dict for round 1
    """
    if cur_round > 1:
        anchor_mask_dict = pickle.load(open(f"{dataset}/anchor_mask_r{cur_round-1}.pkl", "rb"))
    else:
        anchor_mask_dict = {}

    return anchor_mask_dict
