import os
import shutil

def create_speedplus_folder_struc(cfg):

    if os.path.isdir(cfg.AUGMENTATIONS.NEW_DATASET_PATH):
        shutil.rmtree(cfg.AUGMENTATIONS.NEW_DATASET_PATH, ignore_errors = True)
    
    # Create new dataset directory
    os.makedirs(cfg.AUGMENTATIONS.NEW_DATASET_PATH)

    # Create sub-directories
    os.makedirs(cfg.AUGMENTATIONS.NEW_SYNTHETIC_PATH)
    os.makedirs(os.path.join(cfg.AUGMENTATIONS.NEW_SYNTHETIC_PATH, 'images'))

    # Copy camera.json
    shutil.copy(cfg.DATASET.CAMERA_PATH, cfg.AUGMENTATIONS.NEW_CAMERA_PATH)

    # Copy train.json and validation.json files
    shutil.copy(os.path.join(cfg.DATASET.SYNTHETIC_PATH, 'train.json'), os.path.join(cfg.AUGMENTATIONS.NEW_SYNTHETIC_PATH, 'train.json'))
    shutil.copy(os.path.join(cfg.DATASET.SYNTHETIC_PATH, 'validation.json'), os.path.join(cfg.AUGMENTATIONS.NEW_SYNTHETIC_PATH, 'validation.json'))