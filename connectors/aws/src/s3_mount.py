import sys

sys.path.append('D:/01-08-23/ixolerator')
from connectors.aws.s3.config import S3Config

import subprocess
import os


# @class_header(
# description='''
# Class for Amazon S3 mount as local drive''')
class s3_mount():
    def __init__(self, *args, **kwargs):
        self.configs = S3Config()

    '''@method_header(
    description=
    Installs necessary packages of rclone)'''

    def run_script(self):
        config = self.configs.get_s3_configs()

        if os.path.exists(config['installlation_location']):
            return

        cmd = 'mkdir ' + config['installlation_location']
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)

        outfilepath = config['installlation_location'] + config['installation_folder_name'] + '.zip'
        cmd = 'Invoke-WebRequest -Uri ' + config['rclone_download_link'] + ' -OutFile ' + outfilepath
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)

        cmd = 'Expand-Archive -path ' + outfilepath + ' -destinationpath ' + config['installlation_location']
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)

        cmd = 'cp ' + config['installlation_location'] + '/rclone-*-amd64/*.* ' + config['installlation_location']
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)

        cmd = 'choco install winfsp -y'
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)

    # @method_header(
    # description='''
    # Creates mount drive based on the rclone config file values''')
    def create_mount(self):
        config = self.configs.get_s3_configs()

        cmd = 'cp ./connectors/aws/s3/rclone.conf ' + config['installlation_location']
        print("line 45", cmd)
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)

        cmd = config['installlation_location'] + 'rclone mount ' + config['bucket_name'] + ':' + config[
            'bucket_name'] + '/ ' + config['drive_name'] + f":/{config['bucket_name']} --vfs-cache-mode full"
        print("line 48", cmd)
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)

    # @method_header(
    # description='''
    # Terminates mount drive''')
    def terminate(self):
        cmd = 'taskkill /IM rclone.exe /F'
        print("termination", cmd)
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)


if __name__ == '__main__':
    s3_mount = s3_mount()
    s3_mount.run_script()
    s3_mount.create_mount()

