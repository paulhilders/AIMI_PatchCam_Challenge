import torch
import torch.nn as nn
from torchmetrics import AUROC, AUC
from sklearn.metrics import roc_auc_score

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


def auc(logits, targets):
    """
    Computes the area under the receiver operating characteristic curve
    """
    sigmoid = nn.Sigmoid()
    predictions = sigmoid(logits)
    auroc = AUROC()
    return auroc(predictions[:,1], targets)