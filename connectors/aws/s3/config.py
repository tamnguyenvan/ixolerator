from utils.common_defs import *

s3_cfg_windows =\
{
    'drive_name': 'T',
    'bucket_name': 'ixolerator-cloud',
    'rclone_download_link': 'https://downloads.rclone.org/v1.55.1/rclone-v1.55.1-windows-amd64.zip',
    'installation_folder_name': 'rclone1',
    'installlation_location': 'C:/rclone1/'
}

s3_cfg_linux =\
{
    'mount_folder_name': 's3_bucket',
    'bucket_name': 'ixolerator-cloud',
    'access_key_id': 'AKIATWZQTUP5P4QMO43H',
    'secret_access_key': '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',
    'password_file_path': '/home/deeplearningcv/.passwd-s3fs',
    'uid':'1002',
    'gid':'1002'
}


class s3_config_windows():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_s3_configs(self):
        return s3_cfg_windows.copy()


class s3_config_linux():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_s3_configs(self):
        return s3_cfg_linux.copy()

    
if __name__ == '__main__':
    pass