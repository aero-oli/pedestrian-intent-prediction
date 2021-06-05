# Implementation of Trainer

import torch
import numpy as np

import data_loader
from base import BaseTrainer
from torchvision.utils import make_grid
from utils import infinte_loop, MetricTracker

class Trainer(BaseTrainer):
    """
    Class implementation for trainers.
    The class is inherited from the class BaseTrainer.
    """
    def __init__(self, model, criterion, metricFunction, optimizer, configuration, device,
                 dataLoader, validationDataLoader=None, learningRateScheduler=None, epochLength=None):
        """
        Method to initialize an object of type Trainer.

        Parameters
        ----------
        self                    : Trainer
                                  Instance of the class
        model                   : torch.nn.Module
                                  Model to be trained
        criterion               : callable
                                  Criterion to be evaluated (This is usually the loss function to be minimized)
        metricFunction          : callable
                                  Metric functions to evaluate model performance
        optimizer               : torch.optim
                                  Optimizer to be used during training
        device                  : torch.device
                                  Device on which the training would be performed
        dataLoader              : torch.utils.data.DataLoader
                                  Dataset sampler to load training data for model training
        validationDataLoader    : torch.utils.data.DataLoader
                                  Dataset sampler to load validation data for model validation (Default value: None)
        learningRateScheduler   : torch.optim.lr_scheduler
                                  Method to adjust learning rate (Default value: None)
        epochLength             : int
                                  Total number of epochs for training (Default value: None)

        Returns
        -------
        self    : Trainer
                  Initialized object of class Trainer
        """
        # Initialize BaseTrainer class
        super().__init__(model, criterion, metricFunction, optimizer, configuration)

        # Save trainer configuration, device, dataLoaders, learningRateScheduler and loggingStep
        self.configuration = configuration
        self.device = device
        self.dataLoader = dataLoader
        if epochLength is None:
            self.epochLength = len(self.dataLoader)
        else:
            self.dataLoader = infinte_loop(dataLoader)
            self.epochLength = epochLength
        self.validationDataLoader = validationDataLoader
        self.performValidation = (self.validationDataLoader is not None)
        self.learningRateScheduler = learningRateScheduler
        self.loggingStep = int(np.sqrt(dataLoader.batch_size))

        # Set up training and validation metrics
        self.trainingMetrics = MetricTracker("loss", *[individualMetricFunction.__name__ for individualMetricFunction in self.metricFunction], writer=self.writer)
        self.validationMetrics = MetricTracker("loss", *[individualMetricFunction.__name__ for individualMetricFunction in self.metricFunction], writer=self.writer)

    def train_epoch(self, epoch):
        """
        Method to train a single epoch.

        Parameters
        ----------
        self    : Trainer
                  Instance of the class
        epoch   : int
                  Current epoch number

        Returns
        -------
        log     : dict
                  Average of all the metrics in a dictionary
        """
        # Set the model to training mode and start training the model
        self.model.train()
        self.trainingMetrics.reset()
        print(type(self.dataLoader) is data_loader.data_loaders.JaadDataLoader)
        for batchId, (data, target) in enumerate(self.dataLoader):
            print(1)
            data, target = data.to(self.device), target.to(self.device)
            print(2)

            self.optimizer.zero_grad()
            print(3)
            output = self.model(data)
            print(4)
            loss = self.criteria(output, target)
            print(5)
            loss.backward()
            print(6)
            self.optimizer.step()
            print(7)

            # Update training metrics
            self.writer.set_step((epoch - 1)* self.epochLength + batchId)
            print(8)
            self.trainingMetrics.update("loss", loss.item())
            print(9)
            for individualMetric in self.metricFunction:
                self.trainingMetrics.update(individualMetric.__name__, individualMetric(output, target))

            print(10)
            if batchId % self.loggingStep == 0:
                self.logger.debug("Training Epoch: {} {} Loss: {}".format(epoch, self.progress(batchId), loss.item()))
                self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

            print(11)
            if batchId == self.epochLength:
                break

        print(12)
        log = self.trainingMetrics.result()

        print(13)
        if self.performValidation:
            validationLog = self.validate_epoch(epoch)
            log.update(**{"val_"+key: value for key,value in validationLog.items()})

        print(14)
        if self.learningRateScheduler is not None:
            self.learningRateScheduler.step()

        return log

    def validate_epoch(self, epoch):
        """
        Method to validate a single epoch.

        Parameters
        ----------
        self    : Trainer
                  Instance of the class
        epoch   : int
                  Current epoch number

        Returns
        -------
        log     : dict
                  Average of all the metrics in a dictionary
        """
        # Set the model to evaluation mode and start validating the model
        self.model.eval()
        self.validationMetrics.reset()

        with torch.no_grad():
            for batchId, (data, target) in enumerate(self.validationDataLoader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                # Update training metrics
                self.writer.set_step((epoch - 1) * len(self.validationDataLoader) + batchId, "valid")
                self.validationMetrics.update("loss", loss.item())
                for individualMetric in self.metricFunction:
                    self.validationMetrics.update(individualMetric.__name__, individualMetric(output, target))
                self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

        # Update TensorBoardWriter
        for name, parameter in self.model.named_parameters():
            self.writer.add_histogram(name, parameter, bins="auto")

        return self.validationMetrics.result()

    def progress(self, batchId):
        """
        Method to calculate progress of training or validation.

        Parameters
        ----------
        self    : Trainer
                  Instance of the class
        batchId : int
                  Current batch ID

        Returns
        -------
        progress    : str
                      Amount of progress
        """
        base = "[{}/{} ({:.0f}%)]"

        if hasattr(self.dataLoader, "numberOfSamples"):
            current = batchId * self.dataLoader.batch_size
            total = self.dataLoader.numberOfSamples
        else:
            current = batchId
            total = self.epochLength

        return base.format(current, total, 100.0 * current/total)