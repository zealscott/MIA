import numpy as np


def determine_negative_threshold(scores, true_labels, negative_tolerance):
    """
    Determine optimal negative threshold using reference model scores.
    Start from lowest scores and expand upward while maintaining tolerance requirement.

    Args:
        scores: Numpy array of scores (sorted by sample id)
        true_labels: Ground truth membership labels (0 or 1, aligned with scores)
        negative_tolerance: Maximum number of false negatives allowed for non-member classification

    Returns:
        low_threshold: Threshold for identifying non-members (scores <= low_threshold)
        is_valid: Whether a valid threshold was found
    """
    # print(f"determine negative threshold for this round ...")
    sorted_idx = np.argsort(-scores)  # Sort in descending order
    sorted_scores = scores[sorted_idx]
    sorted_labels = true_labels[sorted_idx]

    # Find negative threshold (for non-members)
    negative_threshold = None
    reversed_scores = sorted_scores[::-1]  # Reverse to start from lowest scores
    reversed_labels = sorted_labels[::-1]
    for i in range(negative_tolerance, len(reversed_scores)):
        false_negatives = np.sum(reversed_labels[:i] == 1)  # Count members misclassified as non-members
        if false_negatives <= negative_tolerance:
            negative_threshold = reversed_scores[i - 1]
        else:
            break

    return negative_threshold


def determine_positive_threshold(scores, true_labels, positive_tolerance):
    """
    Determine optimal high threshold using target model scores.
    Start from highest scores and expand downward while maintaining tolerance requirement.

    Args:
        scores: Numpy array of scores (sorted by sample id)
        true_labels: Ground truth membership labels (0 or 1, aligned with scores)
        positive_tolerance: Maximum number of false positives allowed for member classification

    Returns:
        positive_threshold: Threshold for identifying members (scores >= positive_threshold)
    """
    # print(f"determine positive threshold for this round ...")
    sorted_idx = np.argsort(-scores)  # Sort in descending order
    sorted_scores = scores[sorted_idx]
    sorted_labels = true_labels[sorted_idx]

    # Find positive threshold (for members)
    positive_threshold = None
    for i in range(max(positive_tolerance, 1), len(sorted_scores)):
        false_positives = np.sum(sorted_labels[:i] == 0)  # Count non-members misclassified as members
        if false_positives <= positive_tolerance:
            positive_threshold = sorted_scores[i - 1]
        else:
            break
    return positive_threshold