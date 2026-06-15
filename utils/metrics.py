import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(scores, labels):
    """
    scores: array of shape (n,), prediction scores (probability or logit)
    labels: array of shape (n,), ground truth (0 = bona fide, 1 = spoof)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold