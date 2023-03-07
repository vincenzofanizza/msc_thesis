import pandas as pd

import numpy as np
import sys
import os


def dict_to_csv(dict = None, filepath = None):
    '''
    Save dictionary as .csv using pandas.

    '''
    if dict and filepath:
        df = pd.DataFrame.from_dict(dict)
        df.to_csv(path_or_buf = filepath, index = False)
    else:
        raise ValueError('no dictionary passed or no path specified')

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
