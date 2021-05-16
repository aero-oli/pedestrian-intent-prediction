# Implementation of the class BaseTrainer

import torch
from numpy import Inf
from logger import TensorBoardWriter

class BaseTrainer:
    """
    """
    def __init__(self, model, criteria, metricFunction, optimizer, configuration):
        """
        """
        self.configuration = configuration
        #self.logger = config.

    def train(self):
        """
        """
        #@abstractmethod
        pass
        

    def _train_epoch(self, epoch):
        """
        """
        pass

    def _save_checkpoint(self, epoch, saveBest = False):
        """
        """
        pass

    def _resume_checkpoint(self, resumePath):
        """
        """
        pass

