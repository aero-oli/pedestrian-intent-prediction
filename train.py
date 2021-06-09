# Implementation of Training

import torch
import argparse
import numpy as np
import collections
from trainer import Trainer
import model.loss as lossModule
from utils import prepare_device
import model.metric as metricModule
import torch.nn.functional as F
from parse_config import ConfigParser
# import model.model as architectureModule
import model.social_stgcnn as architectureModule
import data_loader.data_loaders as dataModule
from torch_geometric.data import Data, Batch, DenseDataLoader, DataLoader
import data.datasets.custom_dataset as customDataset

# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.double)
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
np.random.seed(SEED)

def main(configuration):
    """
    Entry point for training the experiment.

    Parameters
    ----------
    configuration   : dict
                      User defined configuration for training the experiment

    Returns
    -------
    None
    """

    print("Getting graph dataset... ")

    dataset = configuration.initialize_object("dataset", customDataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = configuration.initialize_object("model", architectureModule).to(device)
    dataset.to_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("Start training...")
    model.train()
    for idx_data, data in enumerate(dataset):
        print("Trainging Video_{}, Number of frames:{}"
              .format("{}".format(idx_data).zfill(4), len(data)))
        batch = Batch.from_data_list(data)
        optimizer.zero_grad()
        out = model(batch, device)
        y = torch.cat([batch.y.cuda(), torch.ones(size=[out.shape[0]-batch.y.shape[0],
                                                 batch.y.shape[1]], device=device)*2], dim=0)
        print(y.dtype, out.dtype)
        loss = lossModule.binary_cross_entropy_loss(out, y.cuda())
        loss.backward()
        optimizer.step()

        break

    '''
    print("Validation...")
    validationDataLoader = dataLoader.split_validation()


    # Build Model Architecture and print to console
    print("Build Model Architecture and print to console")
    model = configuration.initialize_object("architecture", architectureModule)
    logger.info(model)

    # Prepare for (multi-device) GPU training
    device, deviceIds = prepare_device(configuration["numberOfGpus"])
    model = model.to(device)
    if len(deviceIds) > 1:
        model = torch.nn.DataParallel(model, device_ids = deviceIds)

    # Get function handles of loss and metrics
    criterion = getattr(lossModule, configuration["loss"])
    metrics = [getattr(metricModule, individualMetric) for individualMetric in configuration["metrics"]]

    # Build Optimizer, Learning Rate Scheduler and delete every lines containing lr_scheduler for disabling scheduler
    trainiableParameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = configuration.initialize_object("optimizer", torch.optim, trainiableParameters)
    learningRateScheduler = configuration.initialize_object("learningRateScheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      configuration=configuration,
                      device=device,
                      dataLoader=dataLoader,
                      validationDataLoader=validationDataLoader,
                      learningRateScheduler=learningRateScheduler)

    trainer.train()
    '''

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: None)")


    configuration = ConfigParser.from_args(args)
    main(configuration)
