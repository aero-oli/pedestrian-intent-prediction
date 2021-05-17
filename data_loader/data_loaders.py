# Implementation of Data Loader

from base import BaseDataLoader
from torchvision import datasets, transforms

class MnistDataLoader(BaseDataLoader):
    """
    Class implementation for MnistDataLoader.
    The class is inherited from the class BaseDataLoader
    """
    def __init__(self, dataDirectory, batchSize, shuffle=True, validationSplit=0.0, numberOfWorkers=1, training=True):
        """
        Method to initialize an object of type MnistDataLoader

        Parameters
        ----------
        self            : MnistDataLoader
                          Instance of the class
        dataDirectory   : str
                          Directory where the data must be loaded
        batchSize       : int
                          Number of samples per batch to load
        suffle          : bool
                          Set to True to have data resuffled at very epoch
        validationSplit : int/float
                          Number of samples/Percentage of dataset set as validation
        numberOfWorkers : int
                          Number of subprocesses used for data loading

        Returns
        -------
        self    : MnistDataLoader
                  Initialized object of class MnistDataLoader
        """
        requiredTransformations = transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                    ])
        self.dataDirectory = dataDirectory
        self.dataset = datasets.MNIST(self.dataDirectory, train=training, download=True, transform=requiredTransformations)
        super().__init__(self.dataset, batchSize, shuffle, validationSplit, numberOfWorkers)


class GraphDataLoader(BaseDataLoader):
    """
    Class implementation for GraphDataLoader.
    The class is inherited from the class BaseDataLoader
    """
    def __init__(self, dataDirectory, batchSize, shuffle=True, validationSplit=0.0, numberOfWorkers=1, training=True, modulefocus=True):
        """
        Method to initialize an object of type GraphDataLoader

        Parameters
        ----------
        self            : MnistDataLoader
                          Instance of the class
        dataDirectory   : str
                          Directory where the data must be loaded
        batchSize       : int
                          Number of samples per batch to load
        suffle          : bool
                          Set to True to have data resuffled at very epoch
        validationSplit : int/float
                          Number of samples/Percentage of dataset set as validation
        numberOfWorkers : int
                          Number of subprocesses used for data loading
        modulefocus : bool
                          If the training/testing is only focused on the graph module(True) or end-to-end(False)

        Returns
        -------
        self    : MnistDataLoader
                  Initialized object of class MnistDataLoader
        """
        requiredTransformations = transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                    ])
        self.dataDirectory = dataDirectory

        if modulefocus:
            self.dataset = datasets.jaad(self.dataDirectory, train=training, download=True, transforms=requiredTransformations)
            self.returnValuelabels = {}
            for i, key in self.dataset.keys():
                self.returnValuelabels.update({i: key})

            self.Image_Paths = self.dataset.pop(self.returnValuelabels.popitem())


        else: self.dataset = dataset.MNIST(self.dataDirectory, train=training, download=True, transforms=requiredTransformations)

        super().__init__(self.dataset, batchSize, shuffle, validationSplit, numberOfWorkers)

        # if modulefocus:


if __name__ == '__main__':
    print("Start!!")
    print(data.datasets)
    print(datasets.jaad.JAADDataset)
    print(GraphDataLoader("Test", 2**2))