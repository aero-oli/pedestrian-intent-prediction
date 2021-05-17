# Implementation of the class BaseDataLoader

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    """
    Base class implementation for all data loaders. 
    The class is inherited from nn.Module
    """
    def __init__(self, dataset, batchSize, shuffle, validationSplit, numberOfWorkers, collateFunction=default_collate):
        """
        Method to initialize an object of type BaseDataLoader

        Parameters
        ----------
        self            : BaseDataLoader
                          Instance of the class
        dataset         : Dataset
                          Dataset from which to load the data
        batchSize       : int
                          Number of samples per batch to load
        suffle          : bool
                          Set to True to have data resuffled at very epoch
        validationSplit : int/float
                          Number of samples/Percentage of dataset set as validation
        numberOfWorkers : int
                          Number of subprocesses used for data loading
        collateFunction : callable
                          Function to merge a list of samples to form a mini-bacth of Tensor(s). The default value is default_collate()

        Returns
        -------
        self    : BaseDataLoader
                  Initialized object of class BaseDataLoader
        """
        self.validationSplit = validationSplit
        self.shuffle = shuffle

        self.batchId = 0
        self.numberOfSamples = len(dataset)

        self.sampler, self.validationSampler = self._split_sampler(self.validationSplit)

        self.initialKeywordArguments = {
                                            'dataset': dataset,
                                            'batch_size': batchSize,
                                            'shuffle': self.shuffle,
                                            'collate_fn': collateFunction,
                                            'num_workers': numberOfWorkers
                                        }        

        super().__init__(sampler=self.sampler, **self.initialKeywordArguments)

    def _split_sampler(self, split):
        """
        Method to split samplers for training and validation data.

        Parameters
        ----------
        self    : BaseDataLoader
                  Instance of the class
        split   : int/float
                  Number of samples/Percentage of dataset set as validation

        Returns
        -------
        trainingSampler : SubsetRandomSampler
                          Initialized object of class SubsetRandomSampler
        trainingSampler : SubsetRandomSampler
                          Initialized object of class SubsetRandomSampler
        """
        
        if split == 0.0:
            return None, None
        
        idFull = np.arange(self.numberOfSamples)

        np.random.seed(0)
        np.random.shuffle(idFull)

        if(isinstance(split, int)):
            assert split > 0, "[ERROR] Number of samples is negative!"
            assert split < self.numberOfSamples, "[ERROR] Number of samples larger than the entire dataset!"
            validationLength = split
        else:
            validationLength = int(split * self.numberOfSamples)

        validationIds = idFull[0:validationLength]
        trainingIds = np.delete(idFull, np.arange(0, validationLength))

        trainingSampler = SubsetRandomSampler(trainingIds)
        validationSampler = SubsetRandomSampler(validationIds)

        # Turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.numberOfSamples = len(trainingIds)

        return trainingSampler, validationSampler
        
    def split_validation(self):
        """
        Method to split validation data.

        Parameters
        ----------
        self    : BaseDataLoader
                  Instance of the class

        Returns
        -------
        DataLoader  : DataLoader
                      Initialized object of class DataLoader
        """
        if self.validationSampler is None:
            return None
        else:
            return DataLoader(sampler=self.validationSampler, **self.initialKeywordArguments) #TODO: Could have a problem!