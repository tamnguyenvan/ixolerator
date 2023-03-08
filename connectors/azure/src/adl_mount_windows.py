from connectors.azure.adl.config import adl_config
from utils.common_defs import class_header, method_header

import subprocess
import os

@class_header(
description='''
Class for Azure datalake mount as local drive''')
class adl_mount_windows():
    def __init__(self, *args, **kwargs):
        self.configs = adl_config()
    
    @method_header(
    description='''
    Installs necessary packages''')
    def run_script(self):
       config = self.configs.get_adl_configs_windows()

       if os.path.exists(config['installlation_location']):
           return

       cmd = 'mkdir ' + config['installlation_location']
       subprocess.run(["powershell", "-Command", cmd], capture_output=True)

       outfilepath = config['installlation_location'] + config['installation_folder_name'] + '.zip'
       cmd = 'Invoke-WebRequest -Uri ' + config['rclone_download_link'] + ' -OutFile ' + outfilepath
       subprocess.run(["powershell", "-Command", cmd], capture_output=True)
       
       cmd = 'Expand-Archive -path '+ outfilepath +' -destinationpath ' + config['installlation_location']
       subprocess.run(["powershell", "-Command", cmd], capture_output=True)

       cmd = 'cp ' + config['installlation_location'] + '/rclone-*-amd64/*.* ' + config['installlation_location']
       subprocess.run(["powershell", "-Command", cmd], capture_output=True)

       cmd = 'choco install winfsp -y'
       subprocess.run(["powershell", "-Command", cmd], capture_output=True)

    @method_header(
    description='''
    Creates mount drive based on the rclone config file values''')
    def create_mount(self):
       config = self.configs.get_adl_configs_windows()

       cmd = 'cp ./connectors/azure/adl/rclone.conf ' + config['installlation_location']
       subprocess.run(["powershell", "-Command", cmd], capture_output=True)

       cmd = config['installlation_location'] + 'rclone mount ' + config['blob_name'] + ':' + config['blob_name'] + '/ ' + config['drive_name']+ ': --vfs-cache-mode full --no-console'
       subprocess.run(["powershell", "-Command", cmd], capture_output=True)
           

if __name__ == '__main__':
    adl_mount_windows = adl_mount_windows()
    adl_mount_windows.run_script()
    adl_mount_windows.create_mount()

    


