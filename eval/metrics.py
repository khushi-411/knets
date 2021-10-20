import torch
from torch import nn

def _check_transfer(predictions, labels):
    assert predictions.ndim == 1
    assert labels.ndim == 1
    return predictions.astype(torch.int32), labels.astype(torch.int32)

def accuracy(predictions, labels):
    # https://stackoverflow.com/questions/5142418/what-is-the-use-of-assert-in-python
    assert predictions.shape == labels.shape
    p, l = predictions.astype(torch.int32), labels.astype(torch.int32)
    return torch.where(p == l, 1., 0.).mean()

def roc(logits, labels, num_thresholds):
    assert logits.ndim == 1
    assert labels.ndim == 1
    if labels.dtype != torch.int32:
        labels = labels.astype(torch.int32)
    zeros, ones = torch.zeros_like(logits, dtype=torch.int32), torch.ones_like(logits, dtype=torch.int32)
    roc_data = torch.empty((num_thresholds, 2), dtype=torch.float32)
    for i, threshold in enumerate(torch.linspace(0, 1, num=num_thresholds, endpoint=True, dtype=torch.float32)):
        if threshold == 0:
            roc_data[i, :] = [0, 0]
            continue
        if threshold == 1:
            roc_data[i, :] = [1, 1]
            break
        p = torch.where(logits < threshold, zeros, ones)
        tpr = precision(p, labels)
        fpr = recall(p, labels)
        roc_data[i, :] = [fpr, tpr]
    roc_data = torch.sort(roc_data, axis=0)
    return roc_data

def auc(logits, labels, num_thresholds=200):
    # Area Under the Curve: for binary classifier who outputs only [0,1]
    roc_data = roc(logits, labels, num_thresholds)
    diff = torch.diff(roc_data, axis=0)
    _auc = (diff[:, 0] * diff[:, 1] / 2 + roc_data[:-1, 1] * diff[:, 0]).sum()  # areas of triangles + rectangle
    return _auc

def true_pos(predictions, labels):
    return torch.count_nonzero((predictions == 1) & (labels == 1))

def true_neg(predictions, labels):
    return torch.count_nonzero((predictions == 0) & (labels == 0))

def false_pos(predictions, labels):
    return torch.count_nonzero((predictions == 1) & (labels == 0))

def false_neg(predictions, labels):
    return torch.count_nonzero((predictions == 0) & (labels == 1))

def precision(predictions, labels):
    # TP / (FP + FN) = TP / P
    p, l = predictions, labels
    return true_pos(p, l) / torch.count_nonzero(labels)

def recall(predictions, labels):
    # FP / (FP + TN) = FP / N
    p, l = predictions, labels
    return false_neg(p, l) / (len(labels) - torch.count_nonzero(labels))

def specificity(predictions, labels):
    # TN / (TN + FP) = TN / N
    p, l = predictions, labels
    return true_neg(p, l) / (len(labels) - torch.count_nonzero(labels))

def f1_score(predictions, labels):
    pre = precision(predictions, labels)
    re = recall(predictions, labels)
    return 2/(1/pre + 1/re)
