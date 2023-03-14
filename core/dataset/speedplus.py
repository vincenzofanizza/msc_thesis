'''
Script containing classes and methods for the SPEED+ dataset.

'''
import os
import logging
import numpy as np
import pandas as pd
import torch
import albumentations as A

from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2

from .customed_transforms.randomsunflare import RandomSunFlare
from .customed_transforms.coarsedropout  import CoarseDropout
from core.utils.general import load_image


logger = logging.getLogger(__name__)

    
class ObjDetDataset(Dataset):
    """ 
    Code taken from the SPEED+ baseline repository: https://github.com/tpark94/speedplusbaseline.

    Class that defines the OD dataset object.
    
    << csvfile >> is a path to a .csv file containing the following:
    Path to image:     imagepath,
    Tight RoI Coord.:  xmin, xmax, ymin, ymax     [pix]
    True pose:         q0, q1, q2, q3, t1, t2, t3 [-], [m]
    Keypoint Coord.:   kx1, ky1, ..., kx11, ky11  [pix]

    """
    def __init__(self, cfg, transforms = None, is_train = True):
        self.is_train = is_train
        self.root = cfg.DATASET.ROOT
        self.name = cfg.DATASET.NAME
        self.num_classes   = cfg.MODEL.NUM_CLASSES
        self.num_neighbors = cfg.MODEL.NUM_NEIGHBORS

        if is_train:
            # Source domain - train
            csvfile = os.path.join(self.root, self.name, cfg.TRAIN.DOMAIN, cfg.TRAIN.CSV)
        else:
            csvfile = os.path.join(self.root, self.name, cfg.TEST.DOMAIN, cfg.TEST.CSV)

        logger.info('{} from {}'.format(
            'Training' if is_train else 'Testing', csvfile
        ))

        # Read CSV file
        self.csv = pd.read_csv(csvfile, header = None)

        # Image transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        if index < len(self):
            raise AssertionError('Index range error')

        # Read Image & Bbox
        # TODO: Update indeces with actual csv file configuration
        image_path = os.path.join(self.root, self.name, self.csv.iloc[index, 0])
        data    = load_image(image_path)
        bbox    = np.array(self.csv.iloc[index, 1:5], dtype = np.float32)

        # Data transform
        if self.transforms is not None:
            data = self.transforms(data)

        # Return classes & weights (train) or pose (test)
        if self.is_train:
            attClasses = np.array(self.csv.iloc[index, 12:12 + self.num_neighbors], dtype = np.int32)
            attWeights = np.array(self.csv.iloc[index, 12 + self.num_neighbors:], dtype = np.float32)

            # Classes into n-hot vector
            yClasses = np.zeros(self.num_classes, dtype = np.float32)
            yClasses[attClasses] = 1. / self.num_neighbors

            # Weights into n-hot vector as well
            yWeights = np.zeros(self.num_classes, dtype = np.float32)
            yWeights[attClasses] = attWeights

            return data, torch.from_numpy(yClasses), torch.from_numpy(yWeights)
        else:
            q_gt = np.array(self.csv.iloc[index, 5:9],  dtype=np.float32)
            t_gt = np.array(self.csv.iloc[index, 9:12], dtype=np.float32)
            return data, bbox, torch.from_numpy(q_gt), torch.from_numpy(t_gt)
        
class KeyDetDataset(Dataset):
    """ 
    Code taken from the SPPED+ baseline repository: https://github.com/tpark94/speedplusbaseline.

    Class that defines the KD dataset object.
    
    << csvfile >> is a path to a .csv file containing the following:
    Path to image:     imagepath,
    Tight RoI Coord.:  xmin, xmax, ymin, ymax     [pix]
    True pose:         q0, q1, q2, q3, t1, t2, t3 [-], [m]
    Keypoint Coord.:   kx1, ky1, ..., kx11, ky11  [pix]

    """
    def __init__(self, cfg, transforms = None, is_train = True, load_labels = True):
        self.is_train = is_train
        self.load_labels = load_labels
        self.root = cfg.DATASET.ROOT
        self.name = cfg.DATASET.NAME
        # TODO: include num_keypoints in configuration file
        self.num_keypts  = cfg.NUM_KEYPOINTS

        if is_train:
            # Source domain - train
            csvfile = os.path.join(self.root, self.name, cfg.TRAIN.DOMAIN, cfg.TRAIN.CSV)
        else:
            csvfile = os.path.join(self.root, self.name, cfg.TEST.DOMAIN, cfg.TEST.CSV)

        logger.info('{} from {}'.format(
            'Training' if is_train else 'Testing', csvfile
        ))

        # Read CSV file
        self.csv = pd.read_csv(csvfile, header=None)

        # Image transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        if index < len(self):
            raise AssertionError('Index range error')

        # Read Images & Bbox
        image_path = os.path.join(self.root, self.csv.iloc[index, 0])
        data    = load_image(image_path)
        bbox    = np.array(self.csv.iloc[index, 1:5], dtype = np.float32)

        # Load keypoints
        if self.is_train and self.load_labels:
            keypts = np.array(self.csv.iloc[index, 12:], dtype = np.float32)   # [22,]
            keypts = np.transpose(np.reshape(keypts, (self.num_keypts, 2)))  # [2 x 11]
        else:
            keypts = np.zeros((2, self.num_keypts))

        # Data transform
        if self.transforms is not None:
            data = self.transforms(data)

        # Return keypoints (train) or pose (test)
        if self.is_train:
            if self.load_labels:
                return data, keypts
            else:
                return data
        else:
            q_gt = np.array(self.csv.iloc[index, 5:9],  dtype=np.float32)
            t_gt = np.array(self.csv.iloc[index, 9:12], dtype=np.float32)
            return data, bbox, torch.from_numpy(q_gt), torch.from_numpy(t_gt)

class SpeedplusAugmentCfg:
    '''
    Code taken from the SPNv2 repository: https://github.com/tpark94/spnv2.

    Class that defines the SPEED+ augmentation configuration.
    
    '''
    def __init__(self, p = 0.5, brightness_and_contrast = False, blur = False, noise = False, erasing = False, sun_flare = False):
        self.p = p
        self.apply_brightness_and_contrast = brightness_and_contrast
        self.apply_blur = blur
        self.apply_noise = noise
        self.apply_erasing = erasing
        self.apply_sun_flare = sun_flare

    def build_transforms(self, is_train = True, to_tensor = False, load_labels = True):
        '''
        Build augmentation pipeline using albumentations.

        Modified by the original SPNv2 repo to use an SpeedplusAugmentCfg instance instead of a yacs.config.CfgNode instance.

        Args:
            augment_config (class): instance of the SpeedplusAugmentCfg class containing all relevant parameters about the augmentation pipeline.
            to_tensor (bool): flag indicating whether or not the transform should convert the input image to a torch.Tensor object.
            is_train (bool): flag indicating whether the function is called for training or not.
            load_labels (bool): flag indicating whether the bounding box labels should be included in the transform or not.
        
        Return:
            transforms (class): list of transforms to be applied to SPEED+. 

        Rtype:
            albumentations.Compose

        '''
        transforms = []

        # Add augmentation if training, skip if not
        if is_train:
            if self.apply_brightness_and_contrast:
                transforms += [A.RandomBrightnessContrast(brightness_limit = 0.2,
                                                        contrast_limit = 0.2,
                                                        p = self.p)]
            if self.apply_blur:
                transforms += [A.OneOf(
                    [
                        A.MotionBlur(blur_limit = (3,9)),
                        A.MedianBlur(blur_limit = (3,7)),
                        A.GlassBlur(sigma = 0.5,
                                    max_delta = 2)
                    ], p = self.p
                )]
            if self.apply_noise:
                transforms += [A.OneOf(
                    [
                        A.GaussNoise(var_limit = 40**2), # variance [pix]
                        A.ISONoise(color_shift = (0.1, 0.5),
                                intensity = (0.5, 1.0))
                    ], p = self.p
                )]
            if self.apply_erasing:
                transforms += [CoarseDropout(max_holes = 5,
                                            min_holes = 1,
                                            max_ratio = 0.5,
                                            min_ratio = 0.2,
                                            p = self.p)]
            if self.apply_sun_flare:
                transforms += [RandomSunFlare(num_flare_circles_lower = 1,
                                            num_flare_circles_upper = 10,
                                            p = self.p)]
            # TODO: include other augmentations (style randomisation, haze, stars, streaks, Earth background)
        if to_tensor:
            # Normalize by ImageNet stats, then turn into tensor
            transforms += [A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
                        ToTensorV2()]
        # Compose and return
        if load_labels:
            transforms = A.Compose(
                transforms,
                A.BboxParams(format = 'albumentations',       # [xmin, ymin, xmax, ymax] (normalized)
                            label_fields = ['class_labels'])  # Placeholder
            )
        else:
            transforms = A.Compose(
                transforms
            )

        return transform
    
def make_dataloader():