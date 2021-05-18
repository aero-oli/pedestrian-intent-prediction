# Implementation of dataset class for JAAD
import torch
import random
import numpy as np
from torch.utils.data import Dataset
random.seed(0)

class JAADDataset(Dataset):
    """
    Class implementation for JAAD Dataset. 
    The class is inherited from nn.utils.data.Dataset
    """
    def __init__(self, split, isTrain, sequenceLength, datasetPath):
        """
        Split               : float
                                percentage of training/testing data split. The value is the amount of training data
        isTrain             : bool
                                If the dataset should return training data(True) or testing data(False)
        sequenceLength      : float
                                Amount of buffer-frames past pedestrian appearance
        datasetPath         : str
                                percentage of training
        """
        self.split = split
        self.isTrain = isTrain
        self.sequenceLength = sequenceLength
        self.imagePathFormat = datasetPath + r"\{}\{}.jpg"
        print(self.imagePathFormat)

    def __len__(self):
        """
        """
        return len(self.pedstrians)

    def __getitem__(self, idx, frameIdStart = -1):
        """
        """
        pedestrian = self.pedestrian[idx]

        if (frameIdStart != -1):
            frameStart = frameIdStart
        elif (self.isTrain) and (pedestrian['frame_start'] < pedestrian['frame_end'] - self.sequenceLength + 1):
            frameStart = random.randint(pedestrian['frame_start'], pedestrian['frame_end'] - self.sequenceLength + 1)
        else:
            frameStart = self.pedestrian['frame_start']

        if ((self.isTrain) or (frameIdStart != -1)):
            frameIds = [min(pedestrian['frame_end'] - 1, frameStart + i) for i in range(self.sequenceLength)]
        else:
            frameIds = range(pedestrian['frame_start'], pedestrian['frame_end'])

        gtAction = torch.tensor(np.stack([pedestrian['action'][frames] for frames in frameIds]))

        gtBbox = torch.Tensor(np.stack(pedestrian['bbox'][frames].astype(np.float) if len(pedestrian['bbox'][frames]) else np.zeros(4) for frames in frameIds))
        
        objectClass = torch.tensor(np.stack([data['object_class'][frames] for frames in frameIds]))
        
        objectBbox = torch.tensor(np.stack([data['object_bbox'][frames] for frames in frameIds]))

        frames = torch.tensor(np.array(frameIds))

        imagePath = [self.imagePathFormat.format(pedestrian['video'], frames + 1) for frames in frameIds]

        returnValue =   {
                            'GT_Action' : gtAction, 
                            'GT_Bbox' : gtBbox,
                            'Object_Class' : objectClass,
                            'Object_Bbox' : objectBbox,
                            'Frame_Ids' : frames,
                            'Image_Paths' : imagePath,
                        }

        return returnValue