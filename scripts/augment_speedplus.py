'''
Script to generate augmented versions of the SPEED+ dataset.
It generates a baseline augmented dataset and a set of datasets augmented with domain-specific augmentations.

'''
import argparse
import os
import boto3
import shutil

from tqdm import tqdm

import _add_root

from core.dataset import SpeedplusAugmentCfg
from core.run_config import cfg, update_config
from core.utils.aws import load_image_from_s3, save_image_to_s3
from core.utils.dataset import create_speedplus_folder_struc
from core.utils.general import load_image, save_image


def parse_args():
    '''
    Parse arguments from command line.

    '''
    parser = argparse.ArgumentParser(description = 'augment SPEED+')

    parser.add_argument('--cfg',
                        help = 'file for augmentation configuration',
                        required = True,
                        type = str)
    return parser.parse_args()

# TODO: set random seed
def main(cfg):
    '''
    Augment SPEED+ using a configuration object.
    
    '''
    args = parse_args()
    update_config(cfg, args)

    if cfg.PLATFORM != 'local' and cfg.PLATFORM != 'aws':
        raise ValueError('platform selected in configuration file is not supported')

    if cfg.PLATFORM == 'local':
        create_speedplus_folder_struc(cfg)
        synthetic_image_filenames = os.listdir(os.path.join(cfg.DATASET.SYNTHETIC_PATH, 'images').replace('\\', '/'))
    elif cfg.PLATFORM == 'aws':
        s3 = boto3.resource('s3')
        s3_client = boto3.client('s3')

        synthetic_image_objects = s3_client.list_objects_v2(Bucket = cfg.DATASET.ROOT, Prefix = os.path.join(cfg.DATASET.NAME, 'synthetic', 'images').replace('\\', '/'))
        train_labels_src = {
            'Bucket': cfg.AUGMENTATIONS.NEW_ROOT,
            'Key': os.path.join(cfg.AUGMENTATIONS.NEW_DATASET_NAME, 'synthetic', 'test.json').replace('\\', '/')
        }
        valid_labels_src = {
            'Bucket': cfg.AUGMENTATIONS.NEW_ROOT,
            'Key': os.path.join(cfg.AUGMENTATIONS.NEW_DATASET_NAME, 'synthetic', 'validation.json').replace('\\', '/')
        }
        camera_file_src = {
            'Bucket': cfg.AUGMENTATIONS.NEW_ROOT,
            'Key': os.path.join(cfg.AUGMENTATIONS.NEW_DATASET_NAME, cfg.DATASET.CAMERA_FILE).replace('\\', '/')
        }
        
    # Build transformations
    augment_cfg = SpeedplusAugmentCfg(p = cfg.AUGMENTATIONS.P, 
                                    brightness_and_contrast = cfg.AUGMENTATIONS.BRIGHTNESS_AND_CONTRAST, 
                                    blur = cfg.AUGMENTATIONS.BLUR, 
                                    noise = cfg.AUGMENTATIONS.NOISE,
                                    erasing = cfg.AUGMENTATIONS.ERASING,
                                    sun_flare = cfg.AUGMENTATIONS.SUN_FLARE)
    transforms = augment_cfg.build_transforms(is_train = True, to_tensor = False, load_labels = False)

    if cfg.PLATFORM == 'local':
        print('getting dataset files at the local path: {} ...'.format(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME).replace('\\', '/')))
        for filename, _ in zip(synthetic_image_filenames, tqdm(range(1, len(synthetic_image_filenames) + 1), desc = 'augmenting synthetic images')):
            # Load image
            input_filepath = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, 'synthetic', 'images', filename).replace('\\', '/')
            input_image = load_image(filepath = input_filepath)

            # Apply transformations
            transformed_image = transforms(image = input_image)['image']

            # Save transformed image
            output_filepath = os.path.join(cfg.AUGMENTATIONS.NEW_ROOT, cfg.AUGMENTATIONS.NEW_DATASET_NAME, 'synthetic', 'images', filename).replace('\\', '/')
            save_image(transformed_image, filepath = output_filepath)
        print('synthetic images augmented successfully')

        # Copy train.json and validation.json files
        print('copying synthetic labels to new folder...')
        shutil.copy(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, 'synthetic', 'train.json').replace('\\', '/'), os.path.join(cfg.AUGMENTATIONS.NEW_ROOT, cfg.AUGMENTATIONS.NEW_DATASET_NAME, 'synthetic', 'train.json').replace('\\', '/'))
        shutil.copy(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, 'synthetic', 'validation.json').replace('\\', '/'), os.path.join(cfg.AUGMENTATIONS.NEW_ROOT, cfg.AUGMENTATIONS.NEW_DATASET_NAME, 'synthetic', 'validation.json').replace('\\', '/'))
        print('synthetic labels copied successfully')

        # Copy camera.json
        print('copying camera file to new folder...')
        shutil.copy(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.CAMERA_FILE), os.path.join(cfg.AUGMENTATIONS.NEW_ROOT, cfg.AUGMENTATIONS.NEW_DATASET_NAME, cfg.DATASET.CAMERA_FILE))
        print('camera file copied successfully')
    # TODO: verify aws platform case
    elif cfg.PLATFORM == 'aws':
        print('getting dataset files from the S3 folder: {} ...'.format(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME).replace('\\', '/')))
        for object, _ in zip(synthetic_image_objects, tqdm(range(1, len(synthetic_image_objects) + 1), desc = 'augmenting synthetic images')):
            print(object['Key'])
            # Load image
            # TODO: verify the image is in RGB format and not BGR
            # input_image = load_image_from_s3(cfg.DATASET.ROOT, object['Key'])

            # Apply transformations
            # transformed_image = transforms(image = input_image)['image']

            # Save transformed image
            # filename = os.path.basename(object['Key'])
            # save_image_to_s3(transformed_image, cfg.DATASET.ROOT, os.path.join(cfg.AUGMENTATIONS.NEW_DATASET_NAME, 'synthetic', 'images', filename))
        print('synthetic images augmented successfully')

        # Copy train.json and validation.json files
        print('copying synthetic labels to new folder...')
        # s3.meta.client.copy(train_labels_src, cfg.DATASET.ROOT, os.path.join(cfg.AUGMENTATIONS.NEW_DATASET_NAME, 'synthetic', 'train.json'))
        # s3.meta.client.copy(valid_labels_src, cfg.DATASET.ROOT, os.path.join(cfg.AUGMENTATIONS.NEW_DATASET_NAME, 'synthetic', 'validation.json'))
        print('synthetic labels copied successfully')

        # Copy camera.json
        # print('copying camera file to new folder...')
        # s3.meta.client.copy(camera_file_src, cfg.DATASET.ROOT, os.path.join(cfg.AUGMENTATIONS.NEW_DATASET_NAME, cfg.DATASET.CAMERA_FILE))
        print('camera file copied successfully')


if __name__ == '__main__':
    main(cfg)
