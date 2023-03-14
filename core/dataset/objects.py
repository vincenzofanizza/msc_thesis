'''
Script containing classes and methods for the SPEED+ dataset.

'''
import os
import cv2
import logging
import numpy as np
import pandas as pd
import torch
import albumentations as A

from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2

from customed_transforms.randomsunflare import RandomSunFlare
from customed_transforms.coarsedropout  import CoarseDropout


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
            csv_filepath = os.path.join(self.root, self.name, cfg.TRAIN.DOMAIN, cfg.TRAIN.CSV)
        else:
            csv_filepath = os.path.join(self.root, self.name, cfg.TEST.DOMAIN, cfg.TEST.CSV)

        logger.info('{} from {}'.format(
            'Training' if is_train else 'Testing', csv_filepath
        ))

        # Read CSV file
        self.csv = pd.read_csv(csv_filepath, header = None)

        # Image transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('the index provided is out of range')

        # Read Image & Bbox
        # TODO: Update indeces with actual csv file configuration (if necessary)
        image_path = os.path.join(self.root, self.name, self.csv.iloc[index, 0])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = np.array(self.csv.iloc[index, 1:5], dtype = np.float32)
        keypts  = np.zeros((2, 11))  # dummy

        # Data transform
        image, bbox, _ = self.transforms(image, bbox, keypts)

        # Return image and corresponding label
        item = {
            'Image': image,
            'Bbox': bbox
        }

        return item
        
class KeyDetDataset(Dataset):
    """ 
    Code taken from the SPEED+ baseline repository: https://github.com/tpark94/speedplusbaseline.

    Class that defines the KD dataset object.
    
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
        self.num_keypts  = cfg.NUM_KEYPOINTS

        if is_train:
            # Source domain - train
            csv_filepath = os.path.join(self.root, self.name, cfg.TRAIN.DOMAIN, cfg.TRAIN.CSV)
        else:
            csv_filepath = os.path.join(self.root, self.name, cfg.TEST.DOMAIN, cfg.TEST.CSV)

        logger.info('{} from {}'.format(
            'Training' if is_train else 'Testing', csv_filepath
        ))

        # Read CSV file
        self.csv = pd.read_csv(csv_filepath, header = None)

        # Image transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('the index provided is out of range')

        # Read Images & Bbox
        image_path = os.path.join(self.root, self.csv.iloc[index, 0])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      
        bbox = np.array(self.csv.iloc[index, 1:5], dtype = np.float32)

        # Load keypoints
        keypts = np.array(self.csv.iloc[index, 12:], dtype = np.float32)  # [22,]
        keypts = np.transpose(np.reshape(keypts, (self.num_keypts, 2)))  # [2 x 11]

        # Data transform
        image, bbox, _ = self.transforms(image, bbox, keypts)

        # Return image and corresponding label
        item = {
            'Image': image,
            'Keypts': torch.from_numpy(keypts)
        }

        return item

class SpeedplusAugmentCfg:
    '''
    Code taken from the SPNv2 repository: https://github.com/tpark94/spnv2.

    Class that defines the SPEED+ augmentation configuration.
    
    '''
    def __init__(self, cfg):
        self.p = cfg.AUGMENTATIONS.P
        self.brightness_and_contrast = cfg.AUGMENTATIONS.BRIGHTNESS_AND_CONTRAST
        self.blur = cfg.AUGMENTATIONS.BLUR
        self.noise = cfg.AUGMENTATIONS.NOISE
        self.erasing = cfg.AUGMENTATIONS.ERASING
        self.sun_flare = cfg.AUGMENTATIONS.SUN_FLARE

    def build_transforms(self, is_train = True, load_labels = True):
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
            if self.brightness_and_contrast:
                transforms += [A.RandomBrightnessContrast(brightness_limit = 0.2,
                                                        contrast_limit = 0.2,
                                                        p = self.p)]
            if self.blur:
                transforms += [A.OneOf(
                    [
                        A.MotionBlur(blur_limit = (3,9)),
                        A.MedianBlur(blur_limit = (3,7)),
                        A.GlassBlur(sigma = 0.5,
                                    max_delta = 2)
                    ], p = self.p
                )]
            if self.noise:
                transforms += [A.OneOf(
                    [
                        A.GaussNoise(var_limit = 40**2), # variance [pix]
                        A.ISONoise(color_shift = (0.1, 0.5),
                                intensity = (0.5, 1.0))
                    ], p = self.p
                )]
            if self.erasing:
                transforms += [CoarseDropout(max_holes = 5,
                                            min_holes = 1,
                                            max_ratio = 0.5,
                                            min_ratio = 0.2,
                                            p = self.p)]
            if self.sun_flare:
                transforms += [RandomSunFlare(num_flare_circles_lower = 1,
                                            num_flare_circles_upper = 10,
                                            p = self.p)]
        # TODO: include other augmentations (style randomisation, haze, stars, streaks, Earth background)

        # Normalize by ImageNet stats, then turn into tensor
        transforms += [A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
                    ToTensorV2()]

        # Compose and return
        # TODO: don't we need keypoint labels as well?
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

        return transforms