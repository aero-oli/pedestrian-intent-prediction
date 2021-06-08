# Implementation of loss function

import torch.nn.functional as F

def negative_likelihood_loss(output, target):
    """
    Function to calculate the negative likelihood loss.

    Parameters
    ----------
    output  : multiple
              Output from the Neural Network
    target  : multiple
              Ground Truth Value

    Returns
    -------
    loss    : float
              Negative Likelihood Loss
    """
    return F.nll_loss(output, target)



def binary_cross_entropy_loss(output, target):
    """
    Function to calculate the binary cross entropy loss.

    Parameters
    ----------
    output  : multiple
              Output from the Neural Network
    target  : multiple
              Ground Truth Value

    Returns
    -------
    loss    : float
              Binary Cross Entropy Loss
    """
    return F.binary_cross_entropy(output, target)