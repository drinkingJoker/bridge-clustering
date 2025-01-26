import numpy as np


def __compute_over_axis(cm, axis, zero_div=None):
    diag = np.diag(cm)
    zero_div_mask = cm.sum(axis=axis) == 0
    res = diag / cm.sum(axis=axis).clip(min=1e-4)
    if not zero_div is None:
        res[zero_div_mask] = zero_div
    return res, zero_div_mask


def __compute_recall(cm, zero_div=None):
    return __compute_over_axis(cm, axis=1, zero_div=zero_div)


def compute_recall(cm, zero_div=None):
    return __compute_recall(cm, zero_div)[0]


def __compute_precision(cm, zero_div=None):
    return __compute_over_axis(cm, axis=0, zero_div=zero_div)


def compute_precision(cm, zero_div=None):
    return __compute_precision(cm, zero_div)[0]


def compute_f1(cm, zero_div=None):
    precision, prec_mask = __compute_precision(cm, zero_div=zero_div)
    recall, rec_mask = __compute_recall(cm, zero_div=zero_div)
    res = 2 * precision * recall / (precision + recall).clip(min=1e-4)
    if not zero_div is None:
        mask = prec_mask | rec_mask
        res[mask] = zero_div
    return res
