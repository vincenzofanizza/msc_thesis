'''
Script containing general utilities.

'''
import pandas as pd
import cv2


def dict_to_csv(dict, filepath = None):
    '''
    Save dictionary as .csv using pandas.

    '''
    if filepath:
        df = pd.DataFrame.from_dict(dict)
        df.to_csv(path_or_buf = filepath, index = False)
    else:
        raise ValueError('no path specified')

def dict_from_csv(filepath = None):
    '''
    Load dictionary from .csv using pandas.

    '''
    if filepath: 
        df = pd.read_csv(filepath_or_buffer = filepath)
        dict = df.to_dict(orient = 'list')
        # TODO: return a dict that has lists or arrays as values, not strings.
        return dict
    else:
        raise ValueError('no path or data type specified')

def load_image(filepath = None):
    '''
    Load image using opencv.
    It automatically converts the image from BGR to RGB.

    '''
    if filepath:
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        raise ValueError('no path specified')

def save_image(image, filepath = None):
    '''
    Save image using opencv.
    It automatically converts the image from RGB to BGR for opencv.

    '''
    if filepath:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # NOTE: jpeg quality parameter added to preserve SPEED+ compression size when applying augmentations.
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    else:
        raise ValueError('no path specified')    
