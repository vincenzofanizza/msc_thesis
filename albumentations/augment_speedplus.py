'''
Script to generate augmented versions of the SPEED+ dataset.
It generates a baseline augmented dataset and a set of datasets augmented with domain-specific augmentations.

'''
import cv2
import sys
import os

from customed_transforms.speedplus_augment_cfg import SpeedplusAugmentCfg

sys.path.append('../msc_thesis')
import general_utils as gu


# speedplus_folder = 'C:\\Users\\vinci\\OneDrive\\Desktop\\Thesis\\speedplusv2'
input_image_folder = 'C:\\Users\\vinci\\OneDrive\\Desktop\\Thesis\\input'
output_image_folder = 'C:\\Users\\vinci\\OneDrive\\Desktop\\Thesis\\output'

filename = 'image.jpg'

input_filepath = os.path.join(input_image_folder, filename)
output_filepath = os.path.join(output_image_folder, filename)

# Build transformations
augment_cfg = SpeedplusAugmentCfg()
transforms = augment_cfg.build_transforms(is_train = True, to_tensor = False, load_labels = False)

# Load images
image = gu.load_image(filepath = input_filepath)

# Apply transformations
image = cv2.resize(image, (256, 256))
transformed_image = transforms(image = image)['image']

# Save transformed image
gu.save_image(transformed_image, filepath = output_filepath)