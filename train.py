# Implementation of Training

import torch
import argparse
import numpy as np
import sys
import math
import collections
from trainer import Trainer
import model.loss as lossModule
from utils import prepare_device
import model.metric as metricModule
import torch.nn.functional as F
from parse_config import ConfigParser
import model.social_stgcnn as architectureModule
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

    epoch_range = 5
    savePeriod = 1
    filename = "saved models/Model 1/checkpoint.pth"
    print("Getting graph dataset... ")

    dataset = configuration.initialize_object("dataset", customDataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset.to_device(device)

    model = configuration.initialize_object("model", architectureModule).to(device)
    print("Build Model Architecture and print to console\n: {}".format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lossFunction = torch.nn.NLLLoss()

    trainingDataset, validationDataset = dataset.split_dataset(validationSplit=0.2)

    print("Start training...")
    model.train()
    for idx_data, (video_name, data) in enumerate(trainingDataset.items()):
        print(dataset.get_video_classification_no(video_name))
        sys.stdout.write("\nTrainging {}, Video: {}/{}, Number of frames:{}".format(video_name, idx_data+1, len(trainingDataset.keys()), len(data)))
        for epoch in range(epoch_range):
            
            if epoch_range > 1:
                sys.stdout.write("\nEpoch: {}/{}".format(epoch+1, epoch_range))

            total_loss = 0
            correct = 0
            total = 0
            video_pedestrians = 0

            for time_frame, frame in enumerate(data):
                pedestrians = frame.classification.count(1)
                video_pedestrians += pedestrians
                optimizer.zero_grad()
                output = model(frame.cuda(), pedestrians, device)[[i for i in range(pedestrians)]]
                # print("\nPrediction: {}".format(prediction))
                # print("\nPrediction Shape: {}".format(prediction.size()))
                # print("\nPrediction Type: {}".format(prediction.type()))

                y = torch.cat([frame.y.cuda(), torch.ones(size=[output.shape[0]-frame.y.shape[0], frame.y.shape[1]], device=device)*2], dim=0)[[i for i in range(pedestrians)]].long()
                # print("\nGround Truth: {}".format(y))
                # print("\nGround Truth Shape: {}".format(y.size()))
                # print("\nGround Truth Type: {}".format(y.type()))

                loss = lossFunction(output, y)
                prediction = y.detach().clone()

                if not prediction.nelement() == 0:
                    total_loss += loss
                    loss.backward()
                    optimizer.step()
                    for i in range(output.size()[0]):
                        prediction[i] = torch.argmax(output[i], dim=0)

                correct = correct + torch.sub(prediction, y).numel() - torch.count_nonzero(torch.sub(prediction, y))
                total = total + torch.sub(prediction, y).numel()
            
            accuracy = correct / total
            sys.stdout.write(", Total Loss: {:.4f}, Accuracy: {:.4f}, Pedestrians: {}".format(total_loss, accuracy, video_pedestrians))

    sys.stdout.write("\nSaving Model....")
    torch.save(model.state_dict(), filename)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: None)")


    configuration = ConfigParser.from_args(args)
    main(configuration)


"""
    model.eval()
    correct_each_prediction = [0, 0, 0]
    total_each_prediction = [0, 0, 0]
    print("\nCalculating final accuracy...")
    for idx_video, (_, video) in enumerate(validationDataset.items()):
        sys.stdout.write("\rTesting video {}/{}".format(idx_video+1, len(validationDataset.keys())))
        sys.stdout.flush()
        for idx_frame, frame in enumerate(video):
            pred = torch.round(model(frame, device))
            y = torch.cat([frame.y.cuda(),
                           torch.ones(size=[pred.shape[0]-frame.y.shape[0],
                                            frame.y.shape[1]], device=device)*2], dim=0)
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

"""