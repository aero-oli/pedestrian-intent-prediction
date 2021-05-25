# Implementation of Data Loader
import torch

import utils
from sklearn import preprocessing
import pickle
import numpy as np
from base import BaseDataLoader
from torchvision import datasets, transforms
from torch_geometric.data import Data
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

        # d = self.dataset.__getitem__()
        d = self.dataset.__getitem__()


        self.new_dataset = []
        self.new_videos = []
        self.new_frames = []

        data = Data()
        print(d.keys())
        for video_id, video_value in d.items():
            # video = d.get(list(d.keys())[0])
            width = video_value['width']
            height = video_value['height']
            print(video_value['num_frames'])
            for frame_id, frame_value in video_value['frames'].items():
                node_position = np.empty(shape=4)
                node_appearance = np.empty(shape=25)
                node_attributes = np.empty(shape=12)
                node_behavior = np.empty(shape=6)
                node_ground_truth = np.empty(shape=3)
                for object_id, object_value in frame_value.items():
                    object_node_appearance = np.array([])
                    object_node_attributes = np.array([])
                    object_node_behavior = np.array([])



                    print(object_value.keys())
                    print("behavior keys: {}".format(object_value['behavior'].keys()))
                    # print("attributes keys: {}".format(object_value['attributes'].keys()))
                    # print("appearance keys: {}".format(object_value['appearance'].keys()))
                    print("behavior: {}".format(object_value['behavior']))

                    for object_behavior_id, object_behavior_value in object_value['behavior'].items():
                        object_node_behavior = np.hstack([object_node_behavior, int(object_behavior_value)])
                    for object_appearance_id, object_appearance_value in object_value['appearance'].items():
                        object_node_appearance = np.hstack([object_node_appearance, int(object_appearance_value)])
                    for node_attributes_id, node_attributes_value in object_value['attributes'].items():
                        if not node_attributes_id == 'old_id':
                            object_node_attributes = np.hstack([object_node_attributes, int(node_attributes_value)])

                    node_behavior = np.vstack([node_behavior, object_node_behavior])
                    node_attributes = np.vstack([node_attributes, object_node_attributes])
                    node_appearance = np.vstack([node_appearance, object_node_appearance])
                    node_position = np.vstack([node_position, object_value['bbox']])
                    node_ground_truth = np.vstack([node_ground_truth, np.array(object_value['ground_truth'])])


                # print(np.delete(node_behavior, 0, 0))
                # print(np.delete(node_attributes, 0, 0))
                # print(np.delete(node_appearance, 0, 0))
                node_features = np.delete(np.hstack([node_behavior, node_attributes, node_appearance]), 0, 0)
                data = Data(x=torch.as_tensor(node_features),
                            y=torch.as_tensor(np.delete(node_ground_truth, 0, 0)),
                            pos=torch.as_tensor(np.delete(node_position, 0, 0)),
                            width=width,
                            height=height)
                break
            break
        print(data)

        # frame = frames.get(list(frames.keys())[0])

        # print(frame.keys())
        # # print(type(frame_tensor))
        # data = torch.Tensor([])
        # print(frames_tensor.keys)
        # print(frames.keys())
        # print(frame_tensor.num_nodes)

        # for object_id, object_features in frame.items():
        #     x_object = torch.Tensor([])
        #     x_object = torch.cat((x, object_id), 1)
        #     print(x_object, x_object.size())
        #     # print(val.keys())
        #     # print(val)
        #     print(object_id, object_features)
        #     frame_object = torch.Tensor
        #     if type(object_features) is dict:
        #         print("dict")
        #         # For loop
        #     elif object_id in labels:
        #         print(True)




        # nodes = Data.from_dict(frame)
        #
        # print(nodes.keys)
        # print(nodes[nodes.keys[0]])


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


