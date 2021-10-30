# Implementation of Testing
import math

import torch
import sys
import argparse
import numpy as np
from tqdm import tqdm
import model.loss as lossModule
import model.metric as metricModule
from parse_config import ConfigParser
import data_loader.data_loaders as dataModule
import model.social_stgcnn as architectureModule
import data.datasets.custom_dataset as customDataset
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc, f1_score, accuracy_score

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
    filename = "saved models/Model 3/checkpoint.pth"
    print("Getting graph dataset... ")

    dataset = configuration.initialize_object("dataset", customDataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = configuration.initialize_object("model", architectureModule).to(device)
    dataset.to_device(device)

    trainingDataset, validationDataset = dataset.split_dataset(validationSplit=0.2)
    # validationDataset, trainingDataset = dataset.split_dataset(validationSplit=0.248)

    print("Loading Model {}...".format(filename))
    model.load_state_dict(torch.load(filename))

    print("Start testing...")
    model.eval()
    correct_each_prediction = [0]
    total_each_prediction = [0]
    correct = 0
    total = 0

    print("Total number of train videos: {}".format(len(trainingDataset)))
    print("Total number of test videos: {}".format(len(validationDataset)))

    print("Calculating final accuracy...")
    with torch.no_grad():
        for idx_video, (_, video) in enumerate(validationDataset.items()):
            sys.stdout.write("\rTesting video {}/{}".format(idx_video+1, len(validationDataset.keys())))
            sys.stdout.flush()

            for time_frame, frame in enumerate(video):

                pedestrians = frame.classification.count(1)

                #print("***********************************************************")
                #print("***********************************************************")
                #print("Current Frame: {}".format(time_frame))

                output = model(frame.cuda(), pedestrians, device)
                #print("Model Output: {}".format(output))
                #print("Model Output Shape: {}".format(output.size()))

                #print("frame.y: {}".format(frame.y))
                #y = torch.cat([frame.y.cuda(), torch.ones(size=[output.shape[0]-frame.y.shape[0], frame.y.shape[1]], device=device)*2], dim=0)[[i for i in range(pedestrians)]].long()
                y = frame.y.cuda()[[i for i in range(pedestrians)]][:,1].reshape(pedestrians, 1).long()
                #print("Ground Truth: {}".format(y))
                #print("Ground Truth type: {}".format(type(y)))
                #print("Ground Truth Shape: {}".format(y.size()))

                prediction = y.detach().clone()

                if not prediction.nelement() == 0:
                    for i in range(output.size()[0]):
                        prediction[i] = torch.argmax(output[i], dim=0)

                #print("Model Prediction: {}".format(prediction))
                #print("Model Prediction type: {}".format(type(prediction)))

                #overallGroundTruth.append(y.tolist())
                #overallPrediction.append(prediction.tolist())

                correct += torch.sub(prediction, y).numel() - torch.sub(prediction, y).nonzero().size(0)
                total += torch.sub(prediction, y).numel()

                #comparison = torch.sub(prediction, y)
                for pedestrian_in_frame, pedestrian_prediction in enumerate(prediction):
                    for time_in_frame, time_specific_prediction in enumerate(pedestrian_prediction):
                        if not math.isnan(y[pedestrian_in_frame, time_in_frame]):
                            total_each_prediction[time_in_frame] += 1
                            # print(time_specific_prediction, y[pedestrian_in_frame, time_in_frame])
                            if time_specific_prediction == y[pedestrian_in_frame, time_in_frame]:
                                correct_each_prediction[time_in_frame] += 1

                #correct_each_prediction = [cor_pred + comparison[:, it].numel() -
                #                            torch.count_nonzero(comparison[:, it])
                #                            for it, cor_pred in enumerate(correct_each_prediction)]
                
                #total_each_prediction = [cor_pred + comparison[:, it].numel()
                #                          for it, cor_pred in enumerate(total_each_prediction)]

    #overallGroundTruth = [pedestrianGroundTruth for videoGroundTruth in overallGroundTruth for frameGroundTruth in videoGroundTruth for pedestrianGroundTruth in frameGroundTruth]
    #overallPrediction = [pedestrianPrediction for videoPrediction in overallPrediction for framePrediction in videoPrediction for pedestrianPrediction in framePrediction]

    #print("Overall Ground Truth: {}".format(overallGroundTruth))
    #print("Overall Prediction: {}".format(overallPrediction))

    """
    accuracy = accuracy_score(overallGroundTruth, overallPrediction)
    precisionScore = precision_score(overallGroundTruth, overallPrediction, average='macro')
    recallScore = recall_score(overallGroundTruth, overallPrediction, average='macro')
    f1Score = f1_score(overallGroundTruth, overallPrediction, average='macro')
    #aucScore = auc()

    print("Overall Accuracy: {}".format(accuracy))
    print("Overall Precision Score: {}".format(precisionScore))
    print("Overall Recall Score: {}".format(recallScore))
    print("Overall F1 Score: {}".format(f1Score))
    """
    accuracy = correct / total
    print(accuracy)
    total_predictions = sum(total_each_prediction)
    print(total_predictions)
    correct_predictions = sum(correct_each_prediction)
    total_accuracy = correct_predictions / total_predictions
    print(total_accuracy)
    accuracy_each_prediction = [correct_each_prediction[it] / tot for it, tot in enumerate(total_each_prediction)]

    print('Final accuracy frames: {:.4f}'.format(total_accuracy))
    print('Final accuracy for specific frame prediction: \n 15 frames: {:.4f}'.format(accuracy_each_prediction[0]))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Script to train Graph Neural Network")
    args.add_argument("-c", "--config", default=None, type=str, help="Path to the configuration file (Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="Path to the latest checkpoint (Default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="Index of the GPU used (Default: None)")

    configuration = ConfigParser.from_args(args)
    main(configuration)



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
