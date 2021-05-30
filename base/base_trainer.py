# Implementation of the class BaseTrainer

import torch
from numpy import Inf
from abc import abstractmethod
from logger import TensorBoardWriter

class BaseTrainer:
    """
    Base class implementation for all trainers. 
    """
    def __init__(self, model, criterion, metricFunction, optimizer, userConfiguration):
        """
        Method to initialize an object of type BaseTrainer.

        Parameters
        ----------
        self                : BaseTrainer
                              Instance of the class
        model               : torch.nn.Module
                              Model to be trained
        criterion           : callable
                              Criterion to be evaluated (This is usually the loss function to be minimized)
        metricFunction      : callable
                              Metric functions to evaluate model performance
        optimizer           : torch.optim
                              Optimizer to be used during training
        userConfiguration   : ConfigParser
                              User defined configuration to set up base trainer                                 

        Returns
        -------
        self    : BaseTrainer
                  Initialized object of class BaseTrainer
        """
        # Set up BaseTrainer parameters
        self.configuration = userConfiguration
        self.logger = userConfiguration.get_logger("trainer", userConfiguration["trainer"]["verbosity"])
        self.model = model
        self.criterion = criterion
        self.metricFunction = metricFunction
        self.optimizer = optimizer

        configuredTrainer = userConfiguration["trainer"]
        self.epochs = configuredTrainer["epochs"]
        self.savePeriod = configuredTrainer["savePeriod"]
        self.monitor = configuredTrainer.get("monitor", "off")

        # Configuration to Monitor Model Performance and Save the Best Model
        if self.monitor == "off":
            self.monitoredMode = "off"
            self.monitoredBest = 0
        else:
            self.monitoredMode, self.monitoredMetric = self.monitor.split()
            assert self.monitoredMode in ["min", "max"]

            self.monitoredBest = Inf if self.monitoredMode == "min" else -Inf
            self.earlyStop = configuredTrainer.get("earlyStop", Inf)

            if self.earlyStop <= 0:
                self.earlyStop = Inf

        # Set up start epoch, save directory and TensorBoardWriter
        self.startEpoch = 1
        self.checkpointDirectory = userConfiguration.output_directory        
        self.writer = TensorBoardWriter(userConfiguration.log_directory, self.logger, configuredTrainer["tensorboard"])

        # Resume training in case resume is not None
        if userConfiguration.resume is not None:
            self.resume_checkpoint(userConfiguration.resume)

    @abstractmethod
    def train_epoch(self, epoch):
        """
        Method to train a single epoch.
        The method is abstract. Therefore, the function must be implemented in the subclass.

        Parameters
        ----------
        self    : BaseTrainer
                  Instance of the class
        epoch   : int
                  Current epoch number

        Returns
        -------
        None
        """
        raise NotImplementedError

    def train(self):
        """
        Method to perform training.

        Parameters
        ----------
        self    : BaseTrainer
                  Instance of the class

        Returns
        -------
        None
        """
        # Initialize count for checking no improvement
        notImprovedCount = 0

        # Training loop
        for epoch in range(self.startEpoch, self.epochs + 1):

            # Train individual epoch
            result = self.train_epoch(epoch)

            # Update log
            log = {"epoch": epoch}
            log.update(result)
            for key, value in log.items():
                self.logger.info("{}: {}".format(str(key), value))

            # Find best model and stop training if model does not improve
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

            # Save checkpoint
            if epoch % self.savePeriod == 0:
                self.save_checkpoint(epoch, saveBest=best)


    def save_checkpoint(self, epoch, saveBest = False):
        """
        Method to save training checkpoint.

        Parameters
        ----------
        self        : BaseTrainer
                      Instance of the class
        epoch       : int
                      Current epoch number
        saveBest    : bool
                      Set to true if the best model must be saved

        Returns
        -------
        None
        """
        # Save checkpoints
        architecture = type(self.model).__name__
        state = {
                    "architecture": architecture,
                    "epoch": epoch,
                    "stateDictionary": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "monitoredBest": self.monitoredBest,
                    "configuration": self.configuration
                }
        fileName = str(self.checkpointDirectory / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, fileName)
        self.logger.info("Saving Checkpoint: {} ...".format(fileName))

        # Save best model
        if saveBest:
            bestPath = str(self.checkpointDirectory / "model_Best.pth")
            torch.save(state, bestPath)
            self.logger.info("Saving Current Best Model: model_best.pth...")

    def resume_checkpoint(self, resumePath):
        """
        Method to resume training from a saved checkpoint.

        Parameters
        ----------
        self        : BaseTrainer
                      Instance of the class
        resumePath  : pathlib.Path
                      Path to the checkpoint configuration

        Returns
        -------
        None
        """
        # Load configuration from saved checkpoint
        resumePath = str(resumePath)
        self.logger.info("Loading checkpoint: {} ...".format(resumePath))
        checkpoint = torch.load(resumePath)
        self.startEpoch = checkpoint["epoch"] + 1
        self.monitoredBest = checkpoint["monitoredBest"]

        # Perform sanity check for architecture and optimizer and load the architecture and the optimizer
        if checkpoint["configuration"]["architecture"] != self.configuration["architecture"]:
            self.logger.warning("Warning: Architecture configuration given in the configuration file is different from that of the checkpoint."
                                "This may yield an exception while state dictionary is being loaded.")
        self.model.load_state_dict(checkpoint["stateDictionary"])

        if checkpoint["configuration"]["optimizer"]["type"] != self.configuration["optimizer"]["type"]:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of ceheckpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("Checkpoint loaded. Resuming training from epoch {}".format(self.startEpoch))

