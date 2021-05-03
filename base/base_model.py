# Implementation of the class BaseModel

import numpy as np
import torch.nn as nn
from abc import abstractmethod, ABCMeta

class BaseModel(nn.Module):
    """
    Base class implementation for all models. 
    The class is inherited from nn.Module
    """
    def __str__(self):
        """
        Method to get the string representation of the object.

        Parameters
        ----------
        self    : Instance of the class

        Returns
        -------
        string  : String representation of the class

        """
        modelParameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in modelParameters])

        return super().__str__() + "\nTrainable Parameters are: {}".format(params)

    @abstractmethod
    def forward(self, *inputs):
        """
        Method to perform a forward pass on the neural network.
        The method is abstract. Therefore, the function must be implemented in the subclass.

        Parameters
        ----------
        self    : Instance of the class
        *inputs : Variable number of non-keyword arguments

        Returns
        -------
        None
        """

        raise NotImplementedError