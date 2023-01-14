import platform
from pathlib import Path

import tensorflow as tf
from utils import ret_values
from utils.log import Log
from utils.common_defs import method_header
import sys

log = Log()


__all__ = ['get_optimizer']


@method_header(
    description='''
        Function to get optimizer''',
    arguments='''
        name: select optimizer by default (adam) is selected
        ''',
    returns='''
        tensorflow optimizer
        ''')
def get_optimizer(name='Adam', **kwargs):
    if name == 'Adam':
        return tf.keras.optimizers.Adam(**kwargs)
    else:
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__, f"Unsupported optimizer {name}")
        return ret_values.IXO_RET_NOT_SUPPORTED


# @method_header(description='''Get sync directory from config.
# ''', returns='''Sync dir''')
# def get_sync_dir():
#     os_name = platform.system().lower()
#     if 'linux' in os_name:
#         from connectors.aws.s3.config import S3ConfigLinux
#         config = S3ConfigLinux().get_s3_configs()
#         sync_dir = '~/' + Path(config['mount_folder_name'])
#     elif 'windows' in os_name:
#         from connectors.aws.s3.config import S3Config
#         config = S3Config().get_s3_configs()
#         sync_dir = config['drive_name'] + ':/' + config['bucket_name']
#     else:
#         raise ValueError(f'Not supported OS: {os_name}')
#     return sync_dir
