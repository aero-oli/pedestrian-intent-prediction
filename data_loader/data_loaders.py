# Implementation of Data Loader

import pickle
from base import BaseDataLoader
from torchvision import datasets, transforms
import data.datasets.custom_dataset as customDataset

class MnistDataLoader(BaseDataLoader):
    """
    Class implementation for MnistDataLoader.
    The class is inherited from the class BaseDataLoader
    """
    def __init__(self, dataDirectory, batchSize, shuffle=True, validationSplit=0.0, numberOfWorkers=1, training=True):
        """
        Method to initialize an object of type MnistDataLoader

        Parameters
        ----------
        self            : MnistDataLoader
                          Instance of the class
        dataDirectory   : str
                          Directory where the data must be loaded
        batchSize       : int
                          Number of samples per batch to load
        suffle          : bool
                          Set to True to have data resuffled at very epoch
        validationSplit : int/float
                          Number of samples/Percentage of dataset set as validation
        numberOfWorkers : int
                          Number of subprocesses used for data loading
        training        : bool
                          Set to True to have data sampled for training process

        Returns
        -------
        self    : MnistDataLoader
                  Initialized object of class MnistDataLoader
        """
        requiredTransformations = transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))
                                                    ])
        self.dataDirectory = dataDirectory
        self.dataset = datasets.MNIST(self.dataDirectory, train=training, download=True, transform=requiredTransformations)
        super().__init__(self.dataset, batchSize, shuffle, validationSplit, numberOfWorkers)


class JaadDataLoader(BaseDataLoader):
    """
    Class implementation for JaadDataLoader.
    The class is inherited from the class BaseDataLoader
    """
    def __init__(self, annotations, imageDirectoryFormat, batchSize, sequenceLength, prediction, predictionLength, shuffle=True, validationSplit=0.1, numberOfWorkers=1, training=True):
        """
        Method to initialize an object of type JaadDataLoader

        Parameters
        ----------
        self                    : JaadDataLoader
                                  Instance of the class
        annotations             : str
                                  Path to the annotations file
        imageDirectoryFormat    : str
                                  Format of the directory containing images
        batchSize               : int
                                  Number of samples per batch to load
        sequenceLength          : int
                                  Length of the frame sequence for each pedestrian
        prediction              : bool
                                  Set to True to have prediction made in the future
        predictionLength        : int
                                  Length of the prediction frame sequence for each pedestrian
        suffle                  : bool
                                  Set to True to have data resuffled at very epoch
        validationSplit         : int/float
                                  Number of samples/Percentage of dataset set as validation
        numberOfWorkers         : int
                                  Number of subprocesses used for data loading
        training                : bool
                                  Set to True to have data sampled for training process

        Returns
        -------
        self    : JaadDataLoader
                  Initialized object of class JaadDataLoader
        """

        """
        if modulefocus:

            self.dataset, image_Paths = jaad.JAADDataset(split=trainingSplit, isTrain=isTrain, sequenceLength=bufferFrames, datasetPath=self.datasetDir)
            self.labels = self.dataset.__getitem__(0)
            print(self.dataset.__getitem__(0))
            for i, key in self.dataset.__getitem__().keys():
                self.returnValuelabels.update({i: key})

            self.Image_Paths = self.dataset.pop(self.returnValuelabels.popitem())


        else: self.dataset = dataset.MNIST(self.dataDirectory, train=training, download=True, transforms=requiredTransformations)

        super().__init__(self.dataset, batchSize, shuffle, validationSplit, numberOfWorkers)
        """
        self.annotations = annotations

        # print("annotations: {}".format(annotations))
        # print("imageDirectoryFormat: {}".format(imageDirectoryFormat))
        # print("batchSize: {}".format(batchSize))
        # print("sequenceLength: {}".format(sequenceLength))
        # print("prediction: {}".format(prediction))
        # print("predictionLength: {}".format(predictionLength))
        # print("shuffle: {}".format(shuffle))
        # print("validationSplit: {}".format(validationSplit))
        # print("numberOfWorkers: {}".format(numberOfWorkers))
        # print("training: {}".format(training))

        self.dataset = customDataset.JAAD(self.annotations, imageDirectoryFormat, train=training, sequenceLength=sequenceLength, prediction=prediction, predictionLength=predictionLength)


        print("Getting Item from dataset: ")
        # labels
        print(self.dataset.__len__())
        d = self.dataset.__getitem__()
        print("Start Video_converter!!")
        print(len(d.keys()))



        # super().__init__(self.dataset, shuffle, validationSplit, numberOfWorkers, collateFunction=customDataset.collate_jaad)

    # annot_ped_format, is_train, split,
    # seq_len, ped_crop_size, mask_size, collapse_cls,
    # img_path_format, fsegm_format):

    #self.split = opt.split
    #annot_ped = opt.annot_ped_format.format(self.split)
    #with open(annot_ped, 'rb') as handle: self.peds = pickle.load(handle)
    #self.is_train = opt.is_train
    #self.rand_test = opt.rand_test
    #self.seq_len = opt.seq_len
    #self.predict = opt.predict
    #if self.predict: self.all_seq_len = self.seq_len + opt.pred_seq_len
    #else:self.all_seq_len = self.seq_len
    #self.predict_k = opt.predict_k
    #if self.predict_k:self.all_seq_len += self.predict_k
    #self.ped_crop_size = opt.ped_crop_size
    #self.mask_size = opt.mask_size
    #self.collapse_cls = opt.collapse_cls
    #self.combine_method = opt.combine_method
    #self.img_path_format = opt.img_path_format
    #self.driver_act_format = opt.driver_act_format
    #self.fsegm_format = opt.fsegm_format
    #self.save_cache_format = opt.save_cache_format
    #self.load_cache = opt.load_cache
    #self.cache_format = opt.cache_format
    #self.cache_obj_bbox_format = opt.cache_obj_bbox_format
    #self.use_driver = opt.use_driver
    #self.use_pose = opt.use_pose


