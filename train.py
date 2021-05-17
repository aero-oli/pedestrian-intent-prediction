# Implementation of Training

import torch
import argparse
import numpy as np
import collections
from trainer import Trainer
import model.loss as lossModule
from utils import prepare_device
import model.metric as metricModule
from parse_config import ConfigParser
import model.model as architectureModule
import data_loader.data_loaders as dataModule

# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(configuration):
    """
    """
    logger = configuration.get_logger("train")

    # Setup Data loader Instances
    dataLoader = configuration.initialize_object("data_loader", dataModule)
    validationDataLoader = dataLoader.split_validation()

    # Build Model Architecture and print to console
    model = configuration.initialize_object("arch", architectureModule)
    logger.info(model)

    # Prepare for (multi-device) GPU training
    device, deviceIds = prepare_device(configuration["n_gpu"])
    model = model.to(device)
    if len(deviceIds) > 1:
        model = torch.nn.DataParallel(model, device_ids = deviceIds)

    print("device: {}".format(device))
    print("deviceIds: {}".format(deviceIds))

    # Get function handles of loss and metrics
    criterion = getattr(lossModule, configuration["loss"])
    metrics = [getattr(metricModule, met) for met in configuration["metrics"]]

    # Build Optimizer, Learning Rate Scheduler and delete every lines containing lr_scheduler for disabling scheduler
    trainiableParameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = configuration.initialize_object("optimizer", torch.optim, trainiableParameters)
    learningRateScheduler = configuration.initialize_object("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      configuration=configuration,
                      device=device,
                      dataLoader=dataLoader,
                      validationDataLoader=validationDataLoader,
                      learningRateScheduler=learningRateScheduler)

    trainer.train()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: All)")

    """
    # This part of the code is for modifying configuration from the code
    # Currently this is not required for our project

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options =   [
                    CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
                    CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size")
                ]

    config = ConfigParser.from_args(args, options)
    """

    configuration = ConfigParser.from_args(args)
    main(configuration)
