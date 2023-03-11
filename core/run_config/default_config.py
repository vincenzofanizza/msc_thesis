'''
Script containing the default running configuration.

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from yacs.config import CfgNode as CN


_C = CN()

# ------------------------------------------------------------------------------ #
# Basic settings
# ------------------------------------------------------------------------------ #
_C.PLATFORM = 'local'
_C.ROOT ='C:/Users/vinci/OneDrive/Desktop/Thesis'
_C.OUTPUT_DIR = 'C:/Users/vinci/OneDrive/Desktop/Thesis'

# ------------------------------------------------------------------------------ #
# Dataset-related parameters
# ------------------------------------------------------------------------------ #
_C.DATASET = CN()
_C.DATASET.ROOT = 'C:/Users/vinci/OneDrive/Desktop/Thesis'
_C.DATASET.NAME = 'speedplusv2'
_C.DATASET.CAMERA_FILE = 'camera.json'

# ------------------------------------------------------------------------------ #
# Data augmentation
# ------------------------------------------------------------------------------ #
_C.AUGMENTATIONS = CN()
_C.AUGMENTATIONS.NEW_ROOT = 'C:/Users/vinci/OneDrive/Desktop/Thesis'
_C.AUGMENTATIONS.NEW_DATASET_NAME = 'speedplus_augmented_0'
_C.AUGMENTATIONS.P = 0.5
_C.AUGMENTATIONS.BRIGHTNESS_AND_CONTRAST = False
_C.AUGMENTATIONS.BLUR = False
_C.AUGMENTATIONS.NOISE = False
_C.AUGMENTATIONS.ERASING = False
_C.AUGMENTATIONS.SUN_FLARE = False


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)

    # cfg.DATASET.PATH = join(cfg.DATASET.ROOT, cfg.DATASET.NAME).replace('\\', '/')
    # cfg.DATASET.SYNTHETIC_PATH = join(cfg.DATASET.PATH, 'synthetic').replace('\\', '/')
    # cfg.DATASET.CAMERA_PATH = join(cfg.DATASET.PATH, cfg.DATASET.CAMERA_FILE).replace('\\', '/')

    # cfg.AUGMENTATIONS.NEW_DATASET_PATH = join(cfg.AUGMENTATIONS.SAVE_DIR, cfg.AUGMENTATIONS.NEW_DATASET_NAME).replace('\\', '/')
    # cfg.AUGMENTATIONS.NEW_SYNTHETIC_PATH = join(cfg.AUGMENTATIONS.NEW_DATASET_PATH, 'synthetic').replace('\\', '/')
    # cfg.AUGMENTATIONS.NEW_CAMERA_PATH = join(cfg.AUGMENTATIONS.NEW_DATASET_PATH, cfg.DATASET.CAMERA_FILE).replace('\\', '/')

    cfg.freeze()