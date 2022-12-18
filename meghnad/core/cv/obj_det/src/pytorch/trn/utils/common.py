import sys
import os
from typing import Union, List
from pathlib import Path
from utils.common_defs import method_header


@method_header(description='''
               Get meghnad repos's directories''',
               returns='''
               Path to meghnad\' external repos.''')
def get_meghnad_repo_dir() -> Path:
    file_path = Path(os.path.abspath(__file__))
    return file_path.parents[7] / 'repo/obj_det'


def clean_sys_paths(module_name: str):
    new_paths = []
    for path in sys.path:
        if 'yolo' in path and module_name not in path:
            continue
        new_paths.append(path)
    new_paths.append(os.path.join(get_meghnad_repo_dir(), module_name))
    sys.path = new_paths