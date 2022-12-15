import os
from pathlib import Path


def get_meghnad_repo_dir() -> Path:
    file_path = Path(os.path.abspath(__file__))
    return file_path.parents[7] / 'repo/obj_det'
