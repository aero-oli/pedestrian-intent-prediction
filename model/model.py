# Implementation of the neural network model

from base import BaseModel
import torch.nn as neuralNetwork
import torch.nn.functional as function

class MnistModel(BaseModel):
    """
    Class implementation for MnistModel. 
    """
    def __init__(self, numberOfClasses=10):
        """
        Method to initialize an object of type MnistModel.

        Parameters
        ----------
        self            : MnistModel
                          Instance of the class
        numberOfClasses : int
                          Number of output classes (default = 10)

        Returns
        -------
        self    : BaseDataLoader
                  Initialized object of class MnistModel
        """
        super.__init__()
        self.conv1 = neuralNetwork.Conv2d(1, 10, kernel_size=5)
        self.conv2 = neuralNetwork.Conv2d(10, 20, kernel_size=5)
        self.conv2Drop = neuralNetwork.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, numberOfClasses)

    def forward(self, x):
        """
        Method to perform a forward pass on the MnistModel.

        Parameters
        ----------
        self    : MnistModel
                  Instance of the class
        x       : tensor
                  Input to the neural network

        Returns
        -------
        y       : tensor
                  Prediction from the neural network
        """
        x = function.relu(function.max_pool2d(self.conv1(x), 2))
        x = function.relu(function.max_pool2d(self.conv2Drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = function.relu(self.fc1(x))
        x = function.dropout(x, training=self.training)
        x = self.fc2(x)
        return function.log_softmax(x, dim=1)



