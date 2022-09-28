import torch
from torchmetrics import AUROC

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    """
    acc = 0
    for i in range(len(targets)):
        acc += (predictions[i].argmax(dim=-1) == targets[i]).sum()
    accuracy = acc / len(targets)

    return accuracy


def auc(predictions, targets):
    """
    Computes the area under the receiver operating characteristic curve
    """
    auroc = AUROC(pos_label=1)
    return auroc(predictions[:,1], targets)