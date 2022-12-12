from typing import Dict

from .yolov5_train_utils import train as train_yolov5
from .yolov7_train_utils import train as train_yolov7


def get_train_pipeline(arch: str):
    if arch == 'yolov5':
        return train_yolov5
    elif arch == 'yolov7':
        return train_yolov7
