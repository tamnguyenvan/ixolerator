import os
from pathlib import Path
from utils.common_defs import method_header


@method_header(description='''
               Get meghnad repos's directories''',
               returns='''
               Path to meghnad\' external repos.''')
def get_meghnad_repo_dir() -> Path:
    file_path = Path(os.path.abspath(__file__))
    return file_path.parents[7] / 'repo/obj_det'
