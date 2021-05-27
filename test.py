# Implementation of Testing

import torch
import argparse
from tqdm import tqdm
import model.loss as lossModule
import model.metric as metricModule
from parse_config import ConfigParser
import model.model as architectureModule
import data_loader.data_loaders as dataModule

def main(configuration):
    """
    Entry point for testing the experiment.

    Parameters
    ----------
    configuration   : dict
                      User defined configuration for training the experiment

    Returns
    -------
    None
    """
    logger = configuration.get_logger("test")

    # Setup Data loader Instances
    dataLoader = getattr(dataModule, configuration["dataLoader"]["type"])(
                                                                            configuration['dataLoader']['args']['dataDirectory'],
                                                                            batchSize=512,
                                                                            shuffle=False,
                                                                            validationSplit=0.0,
                                                                            training=False,
                                                                            numberOfWorkers=1
                                                                        )

    # Build Model Architecture and print to console
    model = configuration.init_obj("architecture", architectureModule)
    logger.info(model)

    # Get function handles of loss and metrics
    criterion = getattr(lossModule, configuration["loss"])
    metrics = [getattr(metricModule, individualMetric) for individualMetric in configuration["metrics"]]

    # Load saved checkpoint if the testing is resumed from a checkpoint
    logger.info('Loading checkpoint: {} ...'.format(configuration.resume))
    checkpoint = torch.load(configuration.resume)
    stateDictionary = checkpoint["stateDictionary"]
    if configuration["numberOfGpus"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(stateDictionary)

    # Prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Initialize testing metrics and start testing
    totalLoss = 0.0
    totalMetrics = torch.zeros(len(metrics))
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(dataLoader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Save sample images, or do something with output here

            # Compute loss and metrics on test set
            loss = criterion(output, target)
            batchSize = data.shape[0]
            totalLoss += loss.item() * batchSize
            for i, individualMetric in enumerate(metrics):
                totalMetrics[i] += individualMetric(output, target) * batchSize

    # Update log to include loss
    numberOfSamples = len(dataLoader.sampler)
    log = {"loss": totalLoss / numberOfSamples}
    log.update({individualMetric.__name__: totalMetrics[i].item() / numberOfSamples for i, individualMetric in enumerate(metrics)})
    logger.info(log)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: None)")

    configuration = ConfigParser.from_args(args)
    main(configuration)