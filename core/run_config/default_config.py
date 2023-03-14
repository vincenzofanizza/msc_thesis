'''
Script containing the default running configuration.

'''
from yacs.config import CfgNode as CN


_C = CN()

# ------------------------------------------------------------------------------ #
# Basic settings
# ------------------------------------------------------------------------------ #
_C.PLATFORM = 'local'
_C.ROOT ='C:/Users/vinci/OneDrive/Desktop/Thesis'
_C.OUTPUT_DIR = 'C:/Users/vinci/OneDrive/Desktop/Thesis'

# ------------------------------------------------------------------------------ #
# Model settings
# ------------------------------------------------------------------------------ #
_C.MODEL = CN()
_C.MODEL.TYPE = 'KD'
_C.MODEL.NUM_KEYPOINTS = 11
_C.MODEL.NUM_CLASSES = 5000
_C.MODEL.NUM_NEIGHBORS = 5

# ------------------------------------------------------------------------------ #
# Dataset-related parameters
# ------------------------------------------------------------------------------ #
_C.DATASET = CN()
_C.DATASET.ROOT = 'C:/Users/vinci/OneDrive/Desktop/Thesis'
_C.DATASET.NAME = 'speedplusv2'
_C.DATASET.CAMERA_FILE = 'camera.json'
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.DOMAIN = 'synthetic'
_C.DATASET.TRAIN.CSV = 'train.json'
_C.DATASET.VALIDATION = CN()
_C.DATASET.VALIDATION.DOMAIN = 'synthetic'
_C.DATASET.VALIDATION.CSV = 'validation.json'
_C.DATASET.TEST = CN() 
_C.DATASET.TEST.DOMAIN = 'sunlamp'
_C.DATASET.TEST.CSV = 'test.json'

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
    cfg.freeze()