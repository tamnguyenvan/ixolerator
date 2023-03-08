from connectors.aws.s3.config import s3_config_linux
from utils.common_defs import *


import boto3
import os
import shutil
import datetime

@class_header(
description='''
Class for file transfer between Amazon S3 and local drive''')
class s3_utils():
    def __init__(self, *args, **kwargs):
        self.configs = s3_config_linux()
        config = self.configs.get_s3_configs()

        self.s3_client = boto3.client('s3', 
                          aws_access_key_id=config['access_key_id'], 
                          aws_secret_access_key=config['secret_access_key'], 
                          region_name='ap-south-1')

    @method_header(
    description='''
    Uploads file from local drive to Amazon S3''',
    arguments='''
    file_name: Name of the file to be uploaded. 
    bucket_name: Name of the bucket in which the file needs to be uploaded.
    store_as: path of the s3 object where the file needs to be stored''')
    def upload_file_to_bucket(self, file_name:str, bucket_name:str, store_as:str):
        self.s3_client.upload_file(file_name, bucket_name, store_as)

    @method_header(
    description='''
    Downloads file from Amazon S3 to local drive''',
    arguments='''
    bucket_name: Name of the bucket in which the file needs to be uploaded.
    prefix: path of the object in s3
    save_as: path of the local directory where the file needs to be stored''')
    def download_file_from_bucket(self, bucket_name:str, prefix:str, save_as:str):
        for obj in self.s3_client.list_objects(Bucket = bucket_name, Prefix = prefix)['Contents']:

            object_path = save_as + obj['Key'].replace('/','\\')

            if obj['Key'].endswith('/'):
                if not os.path.exists(object_path):
                    os.makedirs(os.path.dirname(object_path))
            else:
                if not os.path.exists(os.path.dirname(object_path)):
                    os.makedirs(os.path.dirname(object_path))
                
                self.s3_client.download_file(bucket_name, obj['Key'], object_path)

    @method_header(
    description='''
    copy file from mounted drive to local drive''',
    arguments='''
    source_path: source location of the file in mounted drive.
    dest_path: destination path where the file needs to be copied.
    is_add_read_per: Its a boolean field, if yes then the copied file permission changes to read only''',
    returns='''
    path where the file has been copied.'''
    )
    def copy_file_from_mounted_drive(self, source_path:str, dest_path:str, is_add_read_per:bool):
        
        try:
            path = shutil.copy(source_path, dest_path)
        except IOError as io_err:
            os.makedirs(os.path.dirname(dest_path))
            path = shutil.copy(source_path, dest_path)

        if os.name != 'nt':
            if is_add_read_per == True:
                self.add_read_permission(dest_path)

        return path
    
    @method_header(
    description='''
    change file permission in local drive''',
    arguments='''
    destination: location where the file permissions need to be changed.
    ''')
    def add_read_permission(self, destination:str):
        if os.path.isdir(destination):
            for filename in os.listdir(destination):
                os.chmod(os.path.join(destination, filename), 0o400)
        elif os.path.isfile(destination):
            os.chmod(destination, 0o400)

    @method_header(
    description='''
    Deletes file from local drive''',
    arguments='''
    filepath: path of the file which needs to be deleted.
    ''')
    def delete_file_from_local_dir(self,filepath:str):
        if os.path.isdir(filepath):  
            shutil.rmtree(filepath)
        elif os.path.isfile(filepath):  
            os.remove(filepath)
         
            
if __name__ == '__main__':
    #s3_utils = s3_utils()

    #source_dir = '/home/deeplearningcv/s3_bucket/'
    #dest_dir = '/home/deeplearningcv/vti_server/'
    #filename = '3775-Article Text-6833-1-10-20190701.pdf'

    #s3_utils.copy_file_from_mounted_drive(source_dir, dest_dir, filename)
    #s3_utils.change_permission(dest_dir)

    pass
    #s3_utils = s3_utils()
    #print(datetime.datetime.now())

    #s3_utils.copy_file_from_mounted_drive('T://Hinglish_model//model_ver_07_12_2022//best_model.joblib', 
    #                                      'C://Users//Souvik//Downloads//best_model.joblib', 
    #                                      True)

   

    #s3_utils.delete_file_from_local_dir('C://Users//Souvik//Downloads//3775-Article Text-6833-1-10-20190701.pdf')

    #bucket_name = 'ixolerator-cloud'
    #prefix = "Hinglish_model/model_ver_07_12_2022/best_model.joblib"
    #save_as = "C:\\Users\\Souvik\\Downloads\\"

    #s3_utils.download_file_from_bucket(bucket_name, prefix, save_as)
    #s3_utils.delete_file_from_local_dir("C:\\Users\\Souvik\\Downloads\\test")

    #print(datetime.datetime.now())
