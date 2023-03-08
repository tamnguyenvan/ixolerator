from connectors.azure.adl.config import adl_config
from utils.common_defs import class_header, method_header

import subprocess
import os

@class_header(
description='''
Class for Azure datalake mount as local drive''')
class adl_mount_linux():
    def __init__(self, *args, **kwargs):
        self.configs = adl_config()

    @method_header(
    description='''
    Installs necessary packages for blobfuse''')    
    def run_script(self):
       config = self.configs.get_adl_configs_linux()

       if os.path.exists(config['mount_folder_name']):
           return

       cmd = 'echo ' + 'accountName ' + config['accountName'] + ' >> ' + config['password_file_path']
       os.system(cmd)

       cmd = 'echo ' + 'accountKey ' + config['accountKey'] + ' >> ' + config['password_file_path']
       os.system(cmd)

       cmd = 'echo ' + 'containerName ' + config['containerName'] + ' >> ' + config['password_file_path']
       os.system(cmd)

       cmd = 'chmod 600 ' + config['password_file_path']
       os.system(cmd)

       cmd = 'mkdir ' + config['mount_folder_name']
       os.system(cmd)

    @method_header(
    description='''
    Creates mount drive based on the config file values''')
    def create_mount(self) -> str:
       config = self.configs.get_adl_configs_linux()

       cmd = 'blobfuse ' + config['mount_folder_name'] + ' --tmp-path=/mnt/resource/blobfusetmp' + ' --config-file=' + config['password_file_path'] + ' -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120'
       os.system(cmd)

       return '~/' + config['mount_folder_name']

if __name__ == '__main__':
    adl_mount_linux = adl_mount_linux()
    adl_mount_linux.run_script()
    adl_mount_linux.create_mount()

    


