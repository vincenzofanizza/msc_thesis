'''
Script to generate augmented versions of the SPEED+ dataset.
It generates a baseline augmented dataset and a set of datasets augmented with domain-specific augmentations.

'''
import cv2
import sys

from customed_transforms.speedplus_augment_cfg import SpeedplusAugmentCfg

sys.path.append('../msc_thesis')
import general_utils as gu


# Define augmentation pipeline
augment_cfg = SpeedplusAugmentCfg()
transforms = augment_cfg.build_transforms(is_train = True, to_tensor = False, load_labels = False)

# Load images
image = gu.load_image(filepath = 'image.jpg')

# Augment images
image = cv2.resize(image, (256, 256))
transformed_image = transforms(image = image)['image']

# Save images
gu.save_image(transformed_image, filepath = 'transformed_image.png')