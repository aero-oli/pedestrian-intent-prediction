# Implementation of the class BaseTrainer

import torch
from numpy import Inf
from abc import abstractmethod
from logger import TensorBoardWriter

class BaseTrainer:
    """
    """
    def __init__(self, model, criteria, metricFunction, optimizer, configuration):
        """
        """
        self.configuration = configuration
        self.logger = configuration.get_logger("trainer", configuration["trainer"]["verbosity"])

        self.model = model
        self.criteria = criteria
        self.metricFunction = metricFunction
        self.optimizer = optimizer

        configuredTrainer = configuration["trainer"]
        self.epochs = configuredTrainer["epochs"]
        self.savePeriod = configuredTrainer["save_period"]
        self.monitor = configuredTrainer.get("monitor", "off")

        # Configuration to Monitor Model Performance and Save the Best Model
        if self.monitor == "off":
            self.monitoredMode = "off"
            self.monitoredBest = 0
        else:
            self.monitoredMode, self.monitoredMetric = self.monitor.split()
            assert self.monitoredMode in ["min", "max"]

            self.monitoredBest = Inf if self.monitoredMode == "min" else -Inf
            self.earlyStop = configuredTrainer.get("early_stop", Inf)

            if self.earlyStop <= 0:
                self.earlyStop = Inf

        self.startEpoch = 1
        self.checkpointDirectory = configuration.saveDirectory
        
        self.writer = TensorBoardWriter(configuration.logDirectory, self.logger, configuredTrainer["tensorboard"])

        if configuration.resume is not None:
            self._resume_checkpoint(configuration.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        """
        raise NotImplementedError

    def train(self):
        """
        """
        notImprovedCount = 0
        for epoch in range(self.startEpoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {"epoch": epoch}
            log.update(result)

            for key, value in log.items():
                self.logger.info("{}: {}".format(str(key), value))

            best = False
            if self.monitoredMode != "off":
                try:
                    improved = (self.monitoredMode == "min" and log[self.monitoredMetric] <= self.monitoredBest) or \
                               (self.monitoredMode == "max" and log[self.monitoredMetric] >= self.monitoredBest)
                except KeyError:
                    self.logger.warning("Warning: Metric {} not found."
                                        "Therefore, model performance monitoring is disabled".format(self.monitoredMetric))
                    self.monitoredMode = "off"
                    improved = False

                if improved:
                    self.monitoredBest = log[self.monitoredMetric]
                    notImprovedCount = 0
                    best = True
                else:
                    notImprovedCount += 1

                if notImprovedCount > self.earlyStop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Therefore, training stops. ".format(self.earlyStop))

            if epoch % self.savePeriod == 0:
                self._save_checkpoint(epoch, saveBest=best)


    def _save_checkpoint(self, epoch, saveBest = False):
        """
        """
        architecture = type(self.model).__name__
        state = {
                    "arch": architecture,
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "monitor_best": self.monitoredBest,
                    "config": self.configuration
                }
        fileName = str(self.checkpointDirectory / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, fileName)
        self.logger.info("Saving Checkpoint: {} ...".format(fileName))
        if saveBest:
            bestPath = str(self.checkpointDirectory / "model_Best.pth")
            torch.save(state, bestPath)
            self.logger.info("Saving Current Best Model: model_best.pth...")

    def _resume_checkpoint(self, resumePath):
        """
        """
        resumePath = str(resumePath)
        self.logger.info("Loading checkpoint: {} ...".format(resumePath))
        checkpoint = torch.load(resumePath)
        self.startEpoch = checkpoint["epoch"] + 1
        self.monitoredBest = checkpoint["monitor_Best"]

        if checkpoint["config"]["arch"] != self.configuration["arch"]:
            self.logger.warning("Warning: Architecture configuration given in the configuration file is different from that of the checkpoint."
                                "This may yield an exception while state dictionary is being loaded.")
        self.model.load_state_dict(checkpoint["state_dict"])

        if checkpoint["config"]["optimizer"]["type"] != self.configuration["optimizer"]["type"]:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of ceheckpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("Checkpoint loaded. Resuming training from epoch {}".format(self.startEpoch))

