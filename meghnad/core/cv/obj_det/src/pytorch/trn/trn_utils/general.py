import os
from pathlib import Path
import platform
from pathlib import Path
from utils.common_defs import method_header


@method_header(description='''
               Get meghnad repos's directories''',
               returns='''
               Path to meghnad\' external repos.''')
def get_meghnad_repo_dir() -> Path:
    file_path = Path(os.path.abspath(__file__))
    return file_path.parents[7] / 'repo/obj_det'

@method_header(description='''Get sync directory from config.
''', returns='''Sync dir''')
def get_sync_dir():
    os_name = platform.system().lower()
    if 'linux' in os_name:
        from connectors.aws.s3.config import S3ConfigLinux
        config = S3ConfigLinux().get_s3_configs()
        sync_dir = '~/' + Path(config['mount_folder_name'])
    elif 'windows' in os_name:
        from connectors.aws.s3.config import S3Config
        config = S3Config().get_s3_configs()
        sync_dir = config['drive_name'] + ':/' + config['bucket_name']
    else:
        raise ValueError(f'Not supported OS: {os_name}')
    return sync_dir