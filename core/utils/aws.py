'''
Script containing aws-related utilities.

'''
import os
import boto3
import cv2
import numpy as np

from tqdm import tqdm
from botocore.exceptions import ClientError


def upload_file_to_s3(filepath, bucket_name, key):
    """
    Upload a file to an S3 bucket.

    Args:
        filepath (str): File to upload.
        bucket_name (str): Bucket to upload to.
        key (str): File path in the S3 bucket.
        
    Return: 
        True if file was uploaded successfully.

    """
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(Filename = filepath, Bucket = bucket_name, Key = key)
    except:
        raise ClientError('file upload was unsuccessful')
    
    return True

def get_all_s3_keys(bucket_name, prefix):
    """
    Get a list of all keys in an S3 bucket containing more than 1,000 objects.
    Note that this function trows a KeyError in case the bucket contains no more than 1,000 objects (see Boto3 documentation for more details).
    
    Args: 
        bucket_name (str): Bucket to retrieve the keys from.
        prefix (str): Key prefix.

    Return:
        bucket keys.
    
    Rtype:
        list
        
    """
    s3_client = boto3.client('s3')

    keys = []
    kwargs = {
        'Bucket': bucket_name,
        'Prefix': prefix
    }

    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            keys.append(obj['Key'])

        print(keys)
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except:
            raise KeyError('no continuation token found. It is likely that the bucket contains no more than 1,000 objects')

    return keys

def save_image_to_s3(image, bucket_name, filepath):
    """
    Save an image into an S3 bucket in jpg format.

    Args:
        image (numpy.array): Array containing the image in RGB format.
        bucket (str): Name of the S3 bucket.

    Return: 
        True if the image was saved successfully.

    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

    s3_client = boto3.client('s3')
    try:
        s3_client.put_object(Body = image_bytes, Bucket = bucket_name, Key = filepath, ContentType = 'image/JPG')
    except:
        raise ClientError('image was not saved successfully')
    
    return True    

def load_image_from_s3(bucket_name, filepath):
    """
    Load image into memory from an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        filepath (str): Image filepath to be loaded into memory.
        
    Return: 
        image in RGB format.
    Rtype:
        numpy.array

    """
    s3_client = boto3.client('s3')

    file_obj = s3_client.get_object(Bucket = bucket_name, Key = filepath)
    file_content = file_obj["Body"].read()
    
    image_bytes = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def upload_speedplus_to_s3(speedplus_root, bucket_name):
    '''
    Upload SPEED+ to an S3 bucket.
    Note that calling this function uploads each file of the SPEED+ dataset to an S3 bucket individually, meaning it takes hours to complete.

    Args:
        speedplus_root (str): Root folder containing the SPEED+ dataset.
        bucket_name (str): Name of the S3 bucket where SPEED+ will be uploaded. 

    Return:
        True if the dataset was uploaded successfully.

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

# TODO: re-write this function
# def unzip_file(): 
#     s3_client=boto3.client('s3')        

#     S3_ZIP_FOLDER = '' 
#     S3_UNZIPPED_FOLDER = 'speedplusv2/' 
#     S3_BUCKET = 'speedplus-dataset' 
#     ZIP_FILE='speedplusv2.zip'     

#     zip_obj = s3_client.get_object(Bucket = S3_BUCKET, Key = f"{S3_ZIP_FOLDER}{ZIP_FILE}") 

#     print("zip_obj=", zip_obj) 

#     buffer = BytesIO(zip_obj["Body"].read()) 
#     z = zipfile.ZipFile(buffer) 

#     # for each file within the zip 
#     for filename in z.namelist(): 

#         # Now copy the files to the 'unzipped' S3 folder 
#         print(f"Copying file {filename} to {S3_BUCKET}/{S3_UNZIPPED_FOLDER}{filename}") 

#         s3_client.put_object(Body = z.open(filename).read(), Bucket = S3_BUCKET, Key = f'{S3_UNZIPPED_FOLDER}{filename}') 
