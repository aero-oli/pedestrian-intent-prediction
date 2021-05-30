# Implementation of random stuff

import json
import torch
import pandas as pd
from torch_geometric.data import Data
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

class MetricTracker:
    """
    Class implementation for tracking all the metrics.
    """
    def __init__(self, *keys, writer=None):
        """
        Method to initialize an object of type MetricTracker.

        Parameters
        ----------
        self    : MetricTracker
                  Instance of the class
        *keys   : tuple
                  Multiple number of non-keyword arguments
        writer  : SummaryWriter
                  Writer to log data for consumption and visualization by TensorBoard

        Returns
        -------
        self    : BaseDataLoader
                  Initialized object of class BaseDataLoader
        """
        self.writer = writer
        self.data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """
        Method to reset the tracked metrics.

        Parameters
        ----------
        self    : MetricTracker
                  Instance of the class

        Returns
        -------
        None
        """
        for columns in self.data.columns:
            self.data[columns].values[:] = 0

    def update(self, key, value, numberOfMetrics=1):
        """
        Method to update the tracked metrics.

        Parameters
        ----------
        self            : MetricTracker
                          Instance of the class
        key             : Multiple
                          Type of metric
        value           : Multiple
                          Value of Metric
        numberOfMetrics : int
                          Number of metrics (Default value: 1)

        Returns
        -------
        None
        """
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.data.total[key] += value * numberOfMetrics
        self.data.counts[key] += numberOfMetrics
        self.data.average[key] = self.data.total[key] / self.data.counts[key]
    
    def average(self, key):
        """
        Method to get the average of the tracked metrics.

        Parameters
        ----------
        self            : MetricTracker
                          Instance of the class
        key             : Multiple
                          Type of metric

        Returns
        -------
        average         : int/float
                          Average of the tracked metrics
        """
        return self.data.average[key]

    def result(self):
        """
        Method to get result of the tracked metrics.
        Currently the result is the average of the tracked metrics.

        Parameters
        ----------
        self        : MetricTracker
                      Instance of the class

        Returns
        -------
        result      : dict
                      Average of all the metrics in a dictionary
        """
        return dict(self.data.average)
    
def ensure_directory(directoryName):
    """
    Function to ensure that a directory exists.
    The function would create a directory if it does not exists.

    Parameters
    ----------
    directoryName   : str
                      Name of the directory

    Returns
    -------
    None
    """
    directoryName = Path(directoryName)
    if not directoryName.is_dir():
        directoryName.mkdir(parents=True, exist_ok=False)

def read_json(fileName):
    """
    Function to read a JSON file.

    Parameters
    ----------
    fileName    : str
                  Name of the JSON file

    Returns
    -------
    content     : OrderedDict
                  JSON file content
    """
    fileName = Path(fileName)
    with fileName.open('r') as jsonFile:
        return json.load(jsonFile, object_hook=OrderedDict)

def write_json(content, fileName):
    """
    Function to write into a JSON file.

    Parameters
    ----------
    content     : str
                  Content to be written into the JSON file
    fileName    : str
                  Name of the JSON file

    Returns
    -------
    None
    """
    fileName = Path(fileName)
    with fileName.open('w') as jsonFile:
        json.dump(content, jsonFile, indent=4, sort_keys=False)

def infinte_loop(dataLoader):
    """
    Wrapper function for endless data loader.

    Parameters
    ----------
    dataLoader  : list
                  Data loader from which the data would be loaded

    Returns
    -------
    data        : Multiple
                  Loaded data
    """
    for loader in repeat(dataLoader):
        yield from loader

def prepare_device(numberOfGpusToBeUsed):
    """
    Function to setup GPU device if available. 
    The function gets the indices of the GPU devices that are used for DataParallel.

    Parameters
    ----------
    numberOfGpusToBeUsed    : int
                              Number of GPUs to be used

    Returns
    -------
    device      : torch.device
                  Type of device available for training
    gpuIdList   : list
                  List of GPUs available for parallel processing
    """
    numberOfGpusAvailable = torch.cuda.device_count()
    if ((numberOfGpusToBeUsed > 0) and (numberOfGpusAvailable == 0)):
        print("[WARNING] There are no GPUs available on this machine. Switching to CPU for training!")
        numberOfGpusToBeUsed = 0

    if (numberOfGpusToBeUsed > numberOfGpusAvailable):
        print("[WARNING] Number of GPUs configured are {} whereas the number of GPUs available are {}! "
                "Switching to {} number of GPUs for training!".format(numberOfGpusToBeUsed, numberOfGpusAvailable, numberOfGpusAvailable))
        numberOfGpusToBeUsed = numberOfGpusAvailable

    device = torch.device('cuda' if numberOfGpusToBeUsed > 0 else 'cpu')
    gpuIdList = list(range(numberOfGpusToBeUsed))

    return device, gpuIdList


def jaad_annotation_converter(dataset):
    '''
    Converts the jaad dataset from the default pedestrian oriented ordet to a frame-by-frame order.

    :param dataset (dict): jaad annotatiosn that need to be converted from default to frame-by-frame

    :return (dict): jaad annotations in on a complete frame-by-frame order
    '''
    prediction_length = [15, 30, 45]

    new_annotations = {}
    for video_name, video in dataset.items():
        video_dict = {}
        video_dict.update({'width': video.get('width')})
        video_dict.update({'height': video.get('height')})
        video_dict.update({'num_frames': video.get('num_frames')})
        all_frames_dict = {}
        for frame_no in range(video.get('num_frames')):
            frame_dict = {}
            for pedestrian_id, pedestrian_value in video.get('ped_annotations').items():
                pedestrian_id_dict = {}
                if frame_no in pedestrian_value.get('frames') and 'behavior' in pedestrian_value.keys() and 'cross' in pedestrian_value.get('behavior',
                                                                                                     {}).keys():
                    frame_index = pedestrian_value.get('frames').index(frame_no)
                    frame_prediction_length = [frame_no + x if (frame_no + x) in pedestrian_value.get('frames') else None for x in prediction_length]
                    for pedestrian_annotation, pedestrian_annotation_value in pedestrian_value.items():
                        if pedestrian_annotation != "frames":
                            if not (type(pedestrian_annotation_value) is dict or type(pedestrian_annotation_value) is list):
                                pedestrian_id_dict.update({pedestrian_annotation: pedestrian_annotation_value})
                            elif type(pedestrian_annotation_value) is list:
                                pedestrian_id_dict.update({pedestrian_annotation: pedestrian_annotation_value[frame_index]})
                            elif type(pedestrian_annotation_value) is dict:
                                pedestiran_id_sub_dict = {}
                                for pedestrian_annotation_sub_id, pedestrian_annotation_sub_value in pedestrian_annotation_value.items():
                                    if pedestrian_annotation_sub_id == 'cross':
                                        frame_prediction_length = [pedestrian_annotation_sub_value[x] if not x is None and x < len(pedestrian_annotation_sub_value) else None for x in frame_prediction_length]
                                    if type(pedestrian_annotation_sub_value) is list:
                                        pedestiran_id_sub_dict.update({pedestrian_annotation_sub_id: pedestrian_annotation_sub_value[frame_index]})
                                    else:
                                        pedestiran_id_sub_dict.update({pedestrian_annotation_sub_id: pedestrian_annotation_sub_value})
                                pedestrian_id_dict.update({pedestrian_annotation: pedestiran_id_sub_dict})
                                pedestrian_id_dict.update({'ground_truth': frame_prediction_length})
                    frame_dict.update({pedestrian_id: pedestrian_id_dict})
                all_frames_dict.update({frame_no: frame_dict})
            # for vehicle_id, vehicle_value in video.get('vehicle_annotations').items():
            #     print("vehicle_value: ",vehicle_value)
            # for traffic_id, traffic_value in video.get('traffic_annotations').items():
            #     print("traffic_value: ",traffic_value)
        video_dict.update({'frames': all_frames_dict})
        new_annotations.update({video_name: video_dict})

    return new_annotations



