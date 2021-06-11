# Implementation of Training

import torch
import argparse
import numpy as np
import sys
import time
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

    epoch_range = 1
    print("Getting graph dataset... ")

    dataset = configuration.initialize_object("dataset", customDataset)
    dataset_limit = len(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = configuration.initialize_object("model", architectureModule).to(device)
    dataset.to_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


    print("Start training...")
    for idx_data, data in zip(range(dataset_limit-2), dataset):
        print("Trainging Video_{}, Number of frames:{}"
              .format("{}".format(idx_data).zfill(4), len(data)))
        model.train()
        for epoch in range(epoch_range):
            sys.stdout.write("Epoch: {}/{}".format(epoch+1, epoch_range))
            total_loss = 0
            correct = 0
            total = 0
            for time_frame, frame in enumerate(data):
                optimizer.zero_grad()
                out = model(frame, device)
                y = torch.cat([frame.y.cuda(), torch.ones(size=[out.shape[0]-frame.y.shape[0],
                                                         frame.y.shape[1]], device=device)*2], dim=0)

                # loss = lossModule.binary_cross_entropy_loss(out, y.cuda())
                loss = torch.mean((out - y) ** 2)
                total_loss += loss
                loss.backward()
                optimizer.step()
                out = torch.round(out)
                correct = correct + torch.sub(out, y).numel() - torch.count_nonzero(torch.sub(out, y))
                total = total + torch.sub(out, y).numel()
            accuracy = correct / total
            sys.stdout.write(", MSE: {}, Accuracy: {} \n".format(total_loss, accuracy))
        model.eval()
        correct = 0
        total = 0
        for frame in dataset[dataset_limit-1]:

            pred = torch.round(model(frame, device))
            y = torch.cat([frame.y.cuda(),
                           torch.ones(size=[pred.shape[0]-frame.y.shape[0],
                                            frame.y.shape[1]], device=device)*2], dim=0)

            correct = correct + torch.sub(pred, y).numel() - torch.count_nonzero(torch.sub(pred, y))
            total = total + torch.sub(pred, y).numel()
        accuracy = correct / total
        print('Final accuracy for video_{}: {:.4f}'.format("{}".format(idx_data).zfill(4), accuracy))

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
