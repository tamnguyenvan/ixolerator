from utils.common_defs import *

adl_cfg_windows =\
{
    'drive_name': 'S',
    'blob_name': 'customertoolsdatastorage',
    'rclone_download_link': 'https://downloads.rclone.org/v1.55.1/rclone-v1.55.1-windows-amd64.zip',
    'installation_folder_name': 'rclone1',
    'installlation_location': 'C:/rclone1/'
}

adl_cfg_linux =\
{
    'containerName': 'customertoolsdatastorage',
    'accountName': 'customertoolsdatastorage',
    'accountKey': 'bm2X3CXkM+cjL2o0oEEwnP8sw3dw7V3e3/iyZABgaKXAruOKprS1k8pk19CRtcSLaaY/eHFndYfv+AStvoj83Q==',
    'password_file_path':'/home/deeplearningcv/fuse_connection.cfg',
    'mount_folder_name':'/home/deeplearningcv/adl'
}


class adl_config():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_adl_configs_windows(self):
        return adl_cfg_windows.copy()

    def get_adl_configs_linux(self):
        return adl_cfg_linux.copy()

    
if __name__ == '__main__':
    pass
