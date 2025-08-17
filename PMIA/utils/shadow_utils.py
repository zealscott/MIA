import os
import numpy as np
import pickle
from typing import Tuple, Dict, List, Union


def mia_output(attack_type, target=True):
    data_type = "target" if target else "shadow"
    if attack_type == "rmia" or attack_type == "entropy":
        return f"softmax_scores_{data_type}.npy"
    elif attack_type == "calibrate" or attack_type == "attackr" or attack_type == "loss":
        return f"losses_{data_type}.npy"
    else:
        return f"scaled_logits_{data_type}.npy"


def get_test(dataset: str, n_shadows: int, attack_type: str = "lira", target=True) -> Tuple[np.ndarray, np.ndarray]:
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
    print(f"load {dataset}/r1_exp/{n_shadows} as test model ...")
    res_file = mia_output(attack_type, target=True)
    test_logits = np.load(f"{dataset}/r1_exp/{n_shadows}/{res_file}")
    test_masks = np.load(f"{dataset}/r1_exp/{n_shadows}/keep.npy")

    return test_logits, test_masks


def get_shadow_res(dataset: str, n_shadow: int, attack_type: str = "lira", target=True) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load the keep mask and scores of the shadow models.

    Args:
        dataset (str): Name of the dataset (e.g., 'cifar10')
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
    shadow_dir = f"{dataset}/r1_exp"

    for model_idx in range(n_shadow):
        model_dir = os.path.join(shadow_dir, str(model_idx))
        shadow_masks.append(np.load(os.path.join(model_dir, "keep.npy")))

        res_file = mia_output(attack_type, target=target)
        shadow_logits.append(np.load(os.path.join(model_dir, res_file)))

    shadow_logits = np.array(shadow_logits)
    shadow_masks = np.array(shadow_masks)

    n_samples = shadow_logits.shape[1]

    return shadow_logits, shadow_masks, n_samples
