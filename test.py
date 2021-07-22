# Implementation of Testing

import torch
import sys
import argparse
from tqdm import tqdm
import model.loss as lossModule
import model.metric as metricModule
from parse_config import ConfigParser
import data_loader.data_loaders as dataModule
import model.social_stgcnn as architectureModule
import data.datasets.custom_dataset as customDataset

torch.set_default_dtype(torch.double)

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
    epoch_range = 1
    savePeriod = 1
    filename = "saved models/Model 2/checkpoint.pth"
    print("Getting graph dataset... ")

    dataset = configuration.initialize_object("dataset", customDataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = configuration.initialize_object("model", architectureModule).to(device)
    dataset.to_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)# , weight_decay=5e-4)

    trainingDataset, validationDataset = dataset.split_dataset(validationSplit=0.2)

    print("Loading Model {}...".format(filename))
    model.load_state_dict(torch.load(filename))

    print("Start testing...")
    model.eval()
    correct_each_prediction = [0, 0, 0]
    total_each_prediction = [0, 0, 0]
    print("Calculating final accuracy...")
    for idx_video, (_, video) in enumerate(validationDataset.items()):
        sys.stdout.write("\rTesting video {}/{}".format(idx_video+1, len(validationDataset.keys())))
        sys.stdout.flush()
        for idx_frame, frame in enumerate(video):
            pedestrians = frame.classification.count(1)
            pred = torch.round(model(frame, device))
            y = torch.cat([frame.y.cuda(),
                           torch.ones(size=[pred.shape[0]-frame.y.shape[0],
                                            frame.y.shape[1]], device=device)*2], dim=0)
            pred = torch.round(pred[[i for i in range(pedestrians)]])
            y = y[[i for i in range(pedestrians)]]

            comparison = torch.sub(pred, y)
            correct_each_prediction = [pred + comparison[:, it].numel() -
                                       torch.count_nonzero(comparison[:, it])
                                       for it, pred in enumerate(correct_each_prediction)]

            total_each_prediction = [pred + comparison[:, it].numel()
                                     for it, pred in enumerate(total_each_prediction)]
            
    total = sum(total_each_prediction)
    correct = sum(correct_each_prediction)
    accuracy = correct / total
    accuracy_each_prediction = [correct_each_prediction[it] / tot
                                for it, tot in enumerate(total_each_prediction)]

    print('Final accuracy frames: {:.4f}'.format(accuracy))
    print('Final accuracy for specific frame prediction: \n '
          '15 frames: {:.4f}, 30 frames: {:.4f}, 45 frames: {:.4f}'
          .format(accuracy_each_prediction[2], accuracy_each_prediction[1], accuracy_each_prediction[0]))

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
    """


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: None)")

    configuration = ConfigParser.from_args(args)
    main(configuration)
