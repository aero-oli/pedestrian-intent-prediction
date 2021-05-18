# Implementation of Data Loader

from base import BaseDataLoader
from torchvision import datasets, transforms
from data.datasets.predetermined import jaad

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
    def __init__(self, datasetDir, bufferFrames=0, batchSize=1, shuffle=True, trainingSplit=0.9, numberOfWorkers=1, isTrain=True, modulefocus=True):
        """
        Method to initialize an object of type GraphDataLoader

        Parameters
        ----------
        self            : MnistDataLoader
                          Instance of the class
        dataDirectory   : str
                          Directory where the data must be loaded
        bufferFrames    : int
                          Number of frames
        batchSize       : int
                          Number of pedestrains sent as a batch
        suffle          : bool
                          Set to True to have data resuffled at very epoch
        trainingSplit   : int/float
                          Number of samples/Percentage of dataset set as training vs testing
        numberOfWorkers : int
                          Number of subprocesses used for data loading
        isTrain : bool
                          If the module should train(True) or test(False)
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
        self.datasetDir = datasetDir

        if modulefocus:

            self.dataset, image_Paths = jaad.JAADDataset(split=trainingSplit, isTrain=isTrain, sequenceLength=bufferFrames, datasetPath=self.datasetDir) #split, isTrain, sequenceLength, imagePathFormat
            self.labels = self.dataset.__getitem__(0)
            print(self.dataset.__getitem__(0))
            for i, key in self.dataset.__getitem__().keys():
                self.returnValuelabels.update({i: key})

            self.Image_Paths = self.dataset.pop(self.returnValuelabels.popitem())


        else: self.dataset = dataset.MNIST(self.dataDirectory, train=training, download=True, transforms=requiredTransformations)

        super().__init__(self.dataset, batchSize, shuffle, validationSplit, numberOfWorkers)

        # if modulefocus:

