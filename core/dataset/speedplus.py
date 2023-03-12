'''
Script containing classes and methods for the SPEED+ dataset.

'''
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

from .customed_transforms.randomsunflare import RandomSunFlare
from .customed_transforms.coarsedropout  import CoarseDropout
from utils.aws import upload_file_to_s3


class SpeedplusAugmentCfg:
    '''
    Code taken from the SPNv2 repository: https://github.com/tpark94/spnv2.

    Class that defines the SPEED+ augmentation configuration.
    
    '''
    def __init__(self, p = 0.5, brightness_and_contrast = False, blur = False, noise = False, erasing = False, sun_flare = False):
        self.p = p
        self.apply_brightness_and_contrast = brightness_and_contrast
        self.apply_blur = blur
        self.apply_noise = noise
        self.apply_erasing = erasing
        self.apply_sun_flare = sun_flare

    def build_transforms(self, is_train = True, to_tensor = False, load_labels = True):
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
            if self.apply_brightness_and_contrast:
                transforms += [A.RandomBrightnessContrast(brightness_limit = 0.2,
                                                        contrast_limit = 0.2,
                                                        p = self.p)]
            if self.apply_blur:
                transforms += [A.OneOf(
                    [
                        A.MotionBlur(blur_limit = (3,9)),
                        A.MedianBlur(blur_limit = (3,7)),
                        A.GlassBlur(sigma = 0.5,
                                    max_delta = 2)
                    ], p = self.p
                )]
            if self.apply_noise:
                transforms += [A.OneOf(
                    [
                        A.GaussNoise(var_limit = 40**2), # variance [pix]
                        A.ISONoise(color_shift = (0.1, 0.5),
                                intensity = (0.5, 1.0))
                    ], p = self.p
                )]
            if self.apply_erasing:
                transforms += [CoarseDropout(max_holes = 5,
                                            min_holes = 1,
                                            max_ratio = 0.5,
                                            min_ratio = 0.2,
                                            p = self.p)]
            if self.apply_sun_flare:
                transforms += [RandomSunFlare(num_flare_circles_lower = 1,
                                            num_flare_circles_upper = 10,
                                            p = self.p)]
            # TODO: include other augmentations (style randomisation, haze, stars, streaks, Earth background)
        if to_tensor:
            # Normalize by ImageNet stats, then turn into tensor
            transforms += [A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
                        ToTensorV2()]
        # Compose and return
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

def upload_speedplus_to_s3(speedplus_root, bucket_name):
    '''
    Upload SPEED+ to an S3 bucket.
    Note that calling this function uploads each file of the SPEED+ dataset to an S3 bucket individually, meaning it takes hours to complete.

    Args:
        speedplus_root (str): Root folder containing the SPEED+ dataset.
        bucket_name (str): Name of the S3 bucket where SPEED+ will be uploaded. 

    Return:
        True if the dataset was uploaded, False otherwise.

    '''
    # # Upload lightbox images with corresponding labels
    # lightbox_path = os.path.join('speedplusv2', 'lightbox').replace('\\', '/')

    # lightbox_image_root = os.path.join(lightbox_path, 'images').replace('\\', '/')
    # lightbox_filenames = os.listdir(os.path.join(speedplus_root, lightbox_image_root).replace('\\', '/'))
    # for image_filename, _ in zip(lightbox_filenames, tqdm(range(1, len(lightbox_filenames) + 1), desc = 'uploading lightbox images')):
    #     upload_file_to_s3(filepath = os.path.join(speedplus_root, lightbox_image_root, image_filename).replace('\\', '/'),
    #                     bucket_name = bucket_name, 
    #                     key = os.path.join(lightbox_image_root, image_filename).replace('\\', '/'))
    # print('lightbox images uploaded successfully')

    # print('uploading lightbox labels...')
    # upload_file_to_s3(filepath = os.path.join(speedplus_root, lightbox_path, 'test.json').replace('\\', '/'), 
    #             bucket_name = bucket_name, 
    #             key = os.path.join(lightbox_path, 'test.json').replace('\\', '/'))
    # print('lightbox labels uploaded successfully')
        
    # # Upload sunlamp images with corresponding labels
    # sunlamp_path = os.path.join('speedplusv2', 'sunlamp').replace('\\', '/')
    
    # sunlamp_image_root = os.path.join(sunlamp_path, 'images').replace('\\', '/')
    # sunlamp_filenames = os.listdir(os.path.join(speedplus_root, sunlamp_image_root).replace('\\', '/'))
    # for image_filename, _ in zip(sunlamp_filenames, tqdm(range(1, len(sunlamp_filenames) + 1), desc = 'uploading sunlamp images')):
    #     upload_file_to_s3(filepath = os.path.join(speedplus_root, sunlamp_image_root, image_filename).replace('\\', '/'),
    #                     bucket_name = bucket_name, 
    #                     key = os.path.join(sunlamp_image_root, image_filename).replace('\\', '/'))
    # print('sunlamp images uploaded successfully')

    # print('uploading sunlamp labels...')
    # upload_file_to_s3(filepath = os.path.join(speedplus_root, sunlamp_path, 'test.json').replace('\\', '/'), 
    #                 bucket_name = bucket_name, 
    #                 key = os.path.join(sunlamp_path, 'test.json').replace('\\', '/'))
    # print('sunlamp labels uploaded successfully')

    # Upload synthetic images with corresponding labels
    synthetic_path = os.path.join('speedplusv2', 'synthetic').replace('\\', '/')
        
    synthetic_image_root = os.path.join(synthetic_path, 'images').replace('\\', '/')
    synthetic_filenames = os.listdir(os.path.join(speedplus_root, synthetic_image_root).replace('\\', '/'))
    for image_filename, _ in zip(synthetic_filenames, tqdm(range(1, len(synthetic_filenames) + 1), desc = 'uploading synthetic images')):
        upload_file_to_s3(filepath = os.path.join(speedplus_root, synthetic_image_root, image_filename).replace('\\', '/'),
                        bucket_name = bucket_name, 
                        key = os.path.join(synthetic_image_root, image_filename).replace('\\', '/'))
    print('synthetic images uploaded successfully')

    print('uploading synthetic labels...')
    upload_file_to_s3(filepath = os.path.join(speedplus_root, synthetic_path, 'train.json').replace('\\', '/'), 
                    bucket_name = bucket_name, 
                    key = os.path.join(synthetic_path, 'train.json').replace('\\', '/'))
    upload_file_to_s3(filepath = os.path.join(speedplus_root, synthetic_path, 'validation.json').replace('\\', '/'), 
                    bucket_name = bucket_name, 
                    key = os.path.join(synthetic_path, 'validation.json').replace('\\', '/'))
    print('synthetic labels updated successfully')

    # Upload camera and license files
    print('uploading camera and license files...')
    upload_file_to_s3(filepath = os.path.join(speedplus_root, 'speedplusv2', 'camera.json').replace('\\', '/'), 
                    bucket_name = bucket_name, 
                    key = os.path.join('speedplusv2', 'camera.json').replace('\\', '/'))
    upload_file_to_s3(filepath = os.path.join(speedplus_root, 'speedplusv2', 'LICENSE.md').replace('\\', '/'), 
                    bucket_name = bucket_name, 
                    key = os.path.join('speedplusv2', 'LICENSE.md').replace('\\', '/'))
    print('camera and license files updated successfully')

    return True