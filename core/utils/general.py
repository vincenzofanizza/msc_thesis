'''
Script containing general utilities.

'''
import pandas as pd
import cv2


def dict_to_csv(dict, filepath):
    '''
    Save dictionary as .csv using pandas.

    '''
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(path_or_buf = filepath, index = False)

def dict_from_csv(filepath):
    '''
    Load dictionary from .csv using pandas.

    '''
    df = pd.read_csv(filepath_or_buffer = filepath)
    dict = df.to_dict(orient = 'list')
    # TODO: return a dict that has lists or arrays as values, not strings.

    return dict

def load_image(filepath):
    '''
    Load image using opencv.
    It automatically converts the image from BGR to RGB.

    '''
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def save_image(image, filepath):
    '''
    Save image using opencv.
    It automatically converts the image from RGB to BGR for opencv.

    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # NOTE: jpeg quality parameter added to preserve SPEED+ compression size when applying augmentations.
    cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 75])
