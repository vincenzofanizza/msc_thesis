'''
Script to generate augmented versions of the SPEED+ dataset.
It generates a baseline augmented dataset and a set of datasets augmented with domain-specific augmentations.

'''
import argparse
import sys
import os

from customed_transforms.speedplus_augment_cfg import SpeedplusAugmentCfg

sys.path.append('../msc_thesis')
import general_utils as gu


def parse_args():
    '''
    Parse arguments from command line.

    '''
    parser = argparse.ArgumentParser(description = 'Augment SPEED+')

    parser.add_argument('--config',
                        help = 'file for augmentation configuration',
                        required = True,
                        type = str)
    return parser.parse_args()

def main(cfg):
    args = parse_args()

# NOTE: should be in the yaml file
speedplus_input_folder = 'C:\\Users\\vinci\\OneDrive\\Desktop\\Thesis\\speedplusv2\\synthetic\\images'
# NOTE: should be in th yaml file
output_folder = 'C:\\Users\\vinci\\OneDrive\\Desktop\\Thesis\\speedplus_augmented\\synthetic\\images'

filenames = os.listdir(speedplus_input_folder)

# Build transformations
# NOTE: augmentation parameters should be in the yaml file
augment_cfg = SpeedplusAugmentCfg(p = 0.5, 
                                brightness_and_contrast = True, 
                                blur = True, 
                                noise = True)
transforms = augment_cfg.build_transforms(is_train = True, to_tensor = False, load_labels = False)

for filename in filenames[:2]:
    print(filename)

    # Create input and output path
    input_filepath = os.path.join(speedplus_input_folder, filename)
    output_filepath = os.path.join(output_folder, filename)
    print(output_filepath)

    # Load images
    input_image = gu.load_image(filepath = input_filepath)

    # Apply transformations
    transformed_image = transforms(image = input_image)['image']

    # Save transformed image
    gu.save_image(transformed_image, filepath = output_filepath)
