import os

import numpy as np
import scipy.stats
import torch
from sklearn.metrics import auc, roc_curve


def compute_metric(score, ground_truth):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(ground_truth, score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def eval_attack(score, ground_truth):
    """
    Evaluate the attack with 10%, 1%, 0.1%, 0.01%, 0.001%, 0.0001% FPR
    """
    print(f"shape of score: {score.shape}, shape of ground_truth: {ground_truth.shape}")

    fpr, tpr, auc, acc = compute_metric(score, ground_truth)

    lw0 = tpr[np.where(fpr < 0.1)[0][-1]]
    lw1 = tpr[np.where(fpr < 0.01)[0][-1]]
    lw2 = tpr[np.where(fpr < 0.001)[0][-1]]
    lw3 = tpr[np.where(fpr < 0.0001)[0][-1]]
    lw4 = tpr[np.where(fpr < 0.00001)[0][-1]]

    return auc, acc, lw0, lw1, lw2, lw3, lw4


def eval_attack_top_x_percent(score, ground_truth, percent):
    """
    Evaluate the attack performance on the top X% of instances with highest absolute scores.
    
    Args:
        score: Attack scores for each instance
        ground_truth: Ground truth membership labels
        percent: Percentage of top instances to select (e.g., 5 for 5%)
        
    Returns:
        auc, acc, lw0, lw1, lw2, lw3, lw4: Evaluation metrics for the selected subset
    """
    print(f"Evaluating attack on top {percent}% of instances with highest absolute scores")
    
    # Calculate absolute scores
    abs_scores = np.abs(score)
    
    # Determine the threshold for the top X%
    threshold = np.percentile(abs_scores, 100 - percent)
    
    # Select instances with absolute scores above the threshold
    selected_indices = np.where(abs_scores >= threshold)[0]
    
    # Get the selected scores and ground truth
    selected_scores = score[selected_indices]
    selected_ground_truth = ground_truth[selected_indices]
    
    print(f"Selected {len(selected_indices)} instances out of {len(score)} ({len(selected_indices)/len(score)*100:.2f}%)")
    
    # Evaluate the attack on the selected subset
    return eval_attack(selected_scores, selected_ground_truth)


@torch.no_grad()
def get_acc(model, dl, device):
    acc = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()
