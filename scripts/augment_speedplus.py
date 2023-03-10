'''
Script to generate augmented versions of the SPEED+ dataset.
It generates a baseline augmented dataset and a set of datasets augmented with domain-specific augmentations.

'''
import argparse
import os

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

def main(cfg):
    '''
    Augment SPEED+ using a configuration object.
    
    '''
    args = parse_args()
    update_config(cfg, args)
    # TODO: adapt this function to S3
    create_speedplus_folder_struc(cfg)

    filenames = os.listdir(os.path.join(cfg.DATASET.SYNTHETIC_PATH, 'images'))

    # Build transformations
    augment_cfg = SpeedplusAugmentCfg(p = cfg.AUGMENTATIONS.P, 
                                    brightness_and_contrast = cfg.AUGMENTATIONS.BRIGHTNESS_AND_CONTRAST, 
                                    blur = cfg.AUGMENTATIONS.BLUR, 
                                    noise = cfg.AUGMENTATIONS.NOISE,
                                    erasing = cfg.AUGMENTATIONS.ERASING,
                                    sun_flare = cfg.AUGMENTATIONS.SUN_FLARE)
    transforms = augment_cfg.build_transforms(is_train = True, to_tensor = False, load_labels = False)

    for filename in filenames[:5]:
        print(filename)

        # Create input and output path
        input_filepath = os.path.join(cfg.DATASET.SYNTHETIC_PATH, 'images', filename)
        output_filepath = os.path.join(cfg.AUGMENTATIONS.NEW_SYNTHETIC_PATH, 'images', filename)

        # Load images
        try:
            input_image = load_image(filepath = input_filepath)
        except:
            input_image = load_image_from_s3(cfg.DATASET.ROOT, os.path.join(cfg.DATASET.NAME, 'synthetic', 'images', filename))

        # Apply transformations
        transformed_image = transforms(image = input_image)['image']

        # Save transformed image
        try:
            save_image(transformed_image, filepath = output_filepath)
        except:
            save_image_to_s3(transformed_image, cfg.DATASET.ROOT, os.path.join(cfg.AUGMENTATIONS.NEW_DATASET_NAME, 'synthetic', 'images', filename))

main(cfg)