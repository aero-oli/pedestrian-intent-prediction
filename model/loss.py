# Implementation of loss function

import torch.nn.functional as function

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
    return function.nll_loss(output, target)



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
    return function.binary_cross_entropy(output, target)