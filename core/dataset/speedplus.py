'''
Script containing classes and methods for the SPEED+ dataset.

'''
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from customed_transforms.randomsunflare import RandomSunFlare
from customed_transforms.coarsedropout  import CoarseDropout


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