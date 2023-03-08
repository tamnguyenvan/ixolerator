from utils.common_defs import *
from connectors.azure.adl.config import adl_config

import os
import shutil
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.storage.blob import ContentSettings, ContainerClient

@class_header(
description='''
Class for file transfer between Azure blob and local drive''')
class adl_utils():

    def __init__(self):
        self.configs = adl_config()
        config = self.configs.get_adl_configs_linux()

        connection_string = "DefaultEndpointsProtocol=https;AccountName=" + config['accountName'] + ";AccountKey=" + config['accountKey']
        self.blob_service_client =  BlobServiceClient.from_connection_string(connection_string)
        self.my_container = self.blob_service_client.get_container_client(config['containerName'])

    @method_header(
    description='''
    Uploads file from local drive to Azure blob''',
    arguments='''
    blob_name: Name of the blob in which the file needs to be uploaded.
    local_path: Path of the file needs to be uploaded.
    container_name: name of the azure blob storage container''')
    def upload_to_blob(self, blob_name:str, local_path:str, container_name:str):
        blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
 
        with open(file=local_path, mode="rb") as data:
            blob_client.upload_blob(data) 
 
    @method_header(
    description='''
    Downloads file from Azure blob to local drive''',
    arguments='''
    blob_name: Name of the blob in which the file needs to be uploaded.
    local_path: Path of the file needs to be uploaded.
    container_name: name of the azure blob storage container''')
    def save_blob(self, blob_name, local_path, container_name):
        local_file_name = blob_name.split('/')[-1]
        download_file_path = os.path.join(local_path, local_file_name)
        container_client = self.blob_service_client.get_container_client(container = container_name) 

        with open(file=download_file_path, mode="wb") as download_file:
            download_file.write(container_client.download_blob(blob_name).readall())

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
    pass
    #azure_blob_file_downloader = azure_blob_file_downloader()

    #blob_name = 'imdb_trn/neg/101_1.txt'
    #local_path = 'D:\\Projects\\Hinglish_model\\model_ver_07_12_2022\\'
    #container_name = 'customertoolsdatastorage'

    #azure_blob_file_downloader.save_blob(blob_name, local_path, container_name)

    #blob_name = 'imdb_trn/neg/test.txt'
    #local_path = 'D:\\Projects\\Hinglish_model\\model_ver_07_12_2022\\test.txt'
    #container_name = 'customertoolsdatastorage'

    #azure_blob_file_downloader.upload_to_blob(blob_name, local_path, container_name)
