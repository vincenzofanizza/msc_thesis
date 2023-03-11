'''
Script containing dataset-related utilities.

'''

import os
import shutil

def create_speedplus_folder_struc(cfg):
    if cfg.PLATFORM != 'local':
        raise ValueError("'local' configuration is required")
    
    if os.path.isdir(cfg.AUGMENTATIONS.NEW_DATASET_PATH):
        shutil.rmtree(cfg.AUGMENTATIONS.NEW_DATASET_PATH, ignore_errors = True)
    
    # Create new dataset directory
    os.makedirs(cfg.AUGMENTATIONS.NEW_DATASET_PATH)

    # Create sub-directories
    os.makedirs(cfg.AUGMENTATIONS.NEW_SYNTHETIC_PATH)
    os.makedirs(os.path.join(cfg.AUGMENTATIONS.NEW_SYNTHETIC_PATH, 'images'))