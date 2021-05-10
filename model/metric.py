# Implementation of metrics required to evaluate training/test output

import torch

def accuracy(output, target):
    """
    Function to calculate the accuracy of the model output.

    Paramaters
    ----------
    output      : multiple
                  Output of the neural network model
    target      : multiple
                  Ground truth value

    Returns
    -------
    accuracy    : float
                  Accuracy of the model output
    """
    with torch.no_grad():
        predictedValue = torch.argmax(output, dim=1)
        assert predictedValue.shape[0] == len(target)
        correct = 0.0
        correct += torch.sum(pred == target).item()

    return (correct/len(target))

def top_k_accuracy(output, target, k=3):
    """
    Function to calculate the top k accuracy of the model output.

    Parameters
    ----------
    output      : multiple
                  Output of the neural network model
    target      : multiple
                  Ground truth value
    k           : int
                  Number of probablities required for the accuracy to be considered

    Returns
    -------
    accuracy    : float
                  Top k accuracy of the model output

    """
    with torch.no_grad():
        predictedValue = torch.topk(output, k, dim=1)[1]
        assert predictedValue.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:,i] == target).item()

    return (correct/len(target))