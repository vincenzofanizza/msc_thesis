import logging
import boto3
import os
import cv2
import numpy as np
import io

from botocore.exceptions import ClientError

from general import save_image


def save_file_to_s3(filepath, bucket):
    """
    Upload a file to an S3 bucket.

    Args:
        filepath (str): File to upload.
        bucket (str): Bucket to upload to.
        
    Return: 
        True if file was uploaded, False otherwise.

    """
    # Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(filepath, bucket)
    except ClientError as e:
        logging.error(e)
        return False
    
    return True

def load_image_from_s3(bucket_name, filepath):
    """
    Load image into memory from an S3 bucket.

    Args:
        file_name (str): Image filename to be loaded into memory.
        bucket (str): Name of the S3 bucket.
        
    Return: 
        image
    Rtype:
        numpy.array

    """
    s3_client = boto3.client('s3')

    file_obj = s3_client.get_object(Bucket = bucket_name, Key = filepath)
    file_content = file_obj["Body"].read()
    
    image_bytes = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    return image

def save_image_to_s3(image, bucket_name, filepath):
    image_bytes = cv2.imencode('.png', image)[1].tobytes()

    s3_client = boto3.client('s3')
    try:
        s3_client.put_object(Body = image_bytes, Bucket = bucket_name, Key = filepath, ContentType = 'image/PNG')
    except ClientError as e:
        logging.error(e)
        return False
    
    return True    
