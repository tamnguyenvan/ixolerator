from connectors.aws.s3.config import S3ConfigLinux
from utils.common_defs import class_header, method_header

import subprocess
import os


@class_header(
    description='''
Class for Amazon S3 mount as local drive''')
class S3MountLinux():
    def __init__(self, *args, **kwargs):
        self.configs = S3ConfigLinux()

    @method_header(
        description='''
    configures directory for mount''')
    def run_script(self):
        config = self.configs.get_s3_configs()

        if os.path.exists(config['mount_folder_name']):
            return

        cmd = 'echo ' + config['access_key_id'] + ':' + config['secret_access_key'] + ' > ' + config[
            'password_file_path']
        print(cmd)
        os.system(cmd)

        cmd = 'chmod 600 ' + config['password_file_path']
        print(cmd)
        os.system(cmd)

        cmd = 'mkdir ' + config['mount_folder_name']
        print(cmd)
        os.system(cmd)

    @method_header(
        description='''
    Creates mount drive using mount folder name specified in config''')
    def create_mount(self):
        config = self.configs.get_s3_configs()

        password_file_path = os.path.expanduser(config['password_file_path'])
        cmd = 's3fs ' + config['bucket_name'] + ':/' + config['mount_folder_name'] + ' ' + \
            config['mount_folder_name'] + \
            ' -o passwd_file=' + password_file_path
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    S3MountLinux = S3MountLinux()
    S3MountLinux.run_script()
    S3MountLinux.create_mount()
