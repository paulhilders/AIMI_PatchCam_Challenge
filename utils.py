import torch

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