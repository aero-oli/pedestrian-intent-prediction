# Implementation of Trainer

import torch
import numpy as np
from base import BaseTrainer
from torchvision.utils import make_grid
from utils import infinte_loop, MetricTracker

class Trainer(BaseTrainer):
    """
    """
    def __init__(self, model, criterion, metricFunction, optimizer, configuration, device,
                 dataLoader, validationDataLoader=None, learningRateScheduler=None, epochLength=None):
        """
        """
        super().__init__(model, criterion, metricFunction, optimizer, configuration)
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

        self.trainingMetrics = MetricTracker("loss", *[m.__name__ for m in self.metricFunction], writer=self.writer)
        self.validationMetrics = MetricTracker("loss", *[m.__name__ for m in self.metricFunction], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        """
        self.model.train()
        self.trainingMetrics.reset()
        for batchId, (data, target) in enumerate(self.dataLoader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1)* self.epochLength + batchId)
            self.trainingMetrics.update("loss", loss.item())
            for individualMetric in self.metricFunction:
                self.trainingMetrics.update(individualMetric.__name__, individualMetric(output, target))

            if batchId % self.loggingStep == 0:
                self.logger.debug("Training Epoch: {} {} Loss: {}".format(epoch, self._progress(batchId), loss.item()))
                self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

            if batchId == self.epochLength:
                break
            
        log = self.trainingMetrics.result()

        if self.performValidation:
            validationLog = self._validate_epoch(epoch)
            log.update(**{"val_"+key: value for key,value in validationLog.items()})

        if self.learningRateScheduler is not None:
            self.learningRateScheduler.step()

        return log

    def _validate_epoch(self, epoch):
        """
        """
        self.model.eval()
        self.validationMetrics.reset()

        with torch.no_grad():
            for batchId, (data, target) in enumerate(self.validationDataLoader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criteria(output, target)

                self.writer.set_step((epoch - 1) * len(self.validationDataLoader) + batchId, "valid")
                self.validationMetrics.update("loss", loss.item())
                for individualMetric in self.metricFunction:
                    self.validationMetrics.update(individualMetric.__name__, individualMetric(output, target))
                self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

        for name, parameter in self.model.named_parameters():
            self.writer.add_histogram(name, parameter, bins="auto")

        return self.validationMetrics.result()

    def _progress(self, batchId):
        """
        """
        base = "[{}/{} ({:.0f}%)]"

        if hasattr(self.dataLoader, "numberOfSamples"):
            current = batchId * self.dataLoader.batch_size
            total = self.dataLoader.numberOfSamples
        else:
            current = batchId
            total = self.epochLength

        return base.format(current, total, 100.0 * current/total)