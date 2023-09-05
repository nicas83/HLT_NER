from collections import defaultdict

import numpy as np
import torch
from pandas import DataFrame
from torch import Tensor, zeros


def scores(confusion: Tensor, all_metrics=False):
    """
    Given a Confusion matrix, returns an F1-score, if all_metrics is false, then returns only a mean of F1-score
    """
    length = confusion.shape[0]
    iter_label = range(length)

    accuracy: Tensor = zeros(length)
    precision: Tensor = zeros(length)
    recall: Tensor = zeros(length)
    f1: Tensor = zeros(length)

    for i in iter_label:
        fn = torch.sum(confusion[i, :i]) + torch.sum(confusion[i, i + 1:])  # false negative
        fp = torch.sum(confusion[:i, i]) + torch.sum(confusion[i + 1:, i])  # false positive
        tn, tp = 0, confusion[i, i]  # true negative, true positive

        for x in iter_label:
            for y in iter_label:
                if (x != i) & (y != i):
                    tn += confusion[x, y]

        accuracy[i] = (tp + tn) / (tp + fn + fp + tn)
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    if all_metrics:
        return DataFrame({
            "Accuracy": accuracy.tolist(),
            "Precision": precision.tolist(),
            "Recall": recall.tolist(),
            "F1": f1.tolist()})
    else:
        return f1.mean()


def f1_score(y_true, y_pred):
    '''
    0,1,2,3 are [CLS],[SEP],[X],O
    '''
    ignore_id = 3

    num_proposed = len(y_pred[y_pred > ignore_id])
    num_correct = (np.logical_and(y_true == y_pred, y_true > ignore_id)).sum()
    num_gold = len(y_true[y_true > ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    return precision, recall, f1


class DictErrors:
    def __init__(self):
        self.e_dict = defaultdict(int)

    def add(self, lst_tokens, lst_pred: Tensor, lst_labels: Tensor):

        for token, pred, lab in zip(lst_tokens, lst_pred, lst_labels):
            if pred != lab:
                self.e_dict[token] += 1

    def result(self) -> dict:
        return dict(sorted(self.e_dict.items(), key=lambda item: item[1], reverse=True))