from connectors.aws.s3.config import s3_config_linux
from utils.common_defs import class_header, method_header

import subprocess
import os

@class_header(
description='''
Class for Amazon S3 mount as local drive''')
class s3_mount_linux():
    def __init__(self, *args, **kwargs):
        self.configs = s3_config_linux()
        
    @method_header(
    description='''
    configures directory for mount''')
    def run_script(self):
       config = self.configs.get_s3_configs()

       if os.path.exists(config['mount_folder_name']):
        return

       cmd = 'echo ' + config['access_key_id'] + ':' + config['secret_access_key']  + ' > ' + config['password_file_path']
       os.system(cmd)

       cmd = 'chmod 600 ' + config['password_file_path']
       os.system(cmd)

       cmd = 'mkdir ' + config['mount_folder_name']
       os.system(cmd)


    @method_header(
    description='''
    Creates mount drive using mount folder name specified in config''',
        returns='''
        extracted text''')
    def create_mount(self) -> str:
       config = self.configs.get_s3_configs()

       cmd = 's3fs ' + config['bucket_name'] + ' ~/' + config['mount_folder_name'] + ' -o passwd_file=' + config['password_file_path'] + ' -o uid=' + config['uid'] + ',gid=' + config['gid']
       os.system(cmd)

       return '~/' + config['mount_folder_name']
           

if __name__ == '__main__':
    s3_mount_linux = s3_mount_linux()
    s3_mount_linux.run_script()
    s3_mount_linux.create_mount()

    


