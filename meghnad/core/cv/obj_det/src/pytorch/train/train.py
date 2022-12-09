import time
import os
import math
import sys
from typing import List, Tuple, Union, Callable, Dict

import numpy as np
import tensorflow as tf

from meghnad.core.cv.obj_det.src.pytorch.data_loader import PytorchObjDetDataLoader
from meghnad.core.cv.obj_det.cfg import ObjDetConfig

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header

from meghnad.core.cv.obj_det.src.pytorch.train.select_model import PytorchObjDetSelectModel
# from meghnad.core.cv.obj_det.src.pytorch.train.eval import PytorchObjDetEval
# from meghnad.core.cv.obj_det.src.pytorch.train.train_utils import get_optimizer
from meghnad.core.cv.obj_det.src.pytorch.train.utils import get_train_pipeline


__all__ = ['TFObjDetTrn']


log = Log()


@method_header(
    description="""Returns configs from given settings

    Parameters
    ----------
    settings : List[str]
        A list of string represents settings.

    Returns
    -------
    [model_cfgs]
        A list of string represents corresponding model configs
    """)
def load_config_from_settings(settings: List[str]) -> Tuple[List, List]:

    settings = [f'{setting}_models' for setting in settings]
    cfg_obj = ObjDetConfig()
    data_cfg = cfg_obj.get_data_cfg()

    model_cfgs = []
    data_cfgs = []
    for setting in settings:
        model_names = cfg_obj.get_model_settings(setting)
        for model_name in model_names:
            model_cfg = cfg_obj.get_model_cfg(model_name)
            model_cfgs.append(model_cfg)
            data_cfgs.append(data_cfg)
    return model_cfgs, data_cfgs


@class_header(
    description='''
        Class for object detection model training''')
class PytorchObjDetTrn:
    def __init__(self, settings: List[str]) -> None:
        self.settings = settings
        self.model_cfgs, self.data_cfgs = load_config_from_settings(settings)
        self.model_selection = PytorchObjDetSelectModel(self.model_cfgs)
        self.data_loaders = []
        self.best_model_path = None

    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data (should point to the file in case of a single file, should point to
                the directory in case data exists in multiple files in a directory structure)
                ''')
    def config_connectors(self, data_path: str, augmentations: Dict = None) -> None:
        self.data_loaders = [PytorchObjDetDataLoader(data_path, data_cfg, model_cfg, augmentations)
                             for data_cfg, model_cfg in zip(self.data_cfgs, self.model_cfgs)]

    @method_header(
        description='''
                Function to set training configurations and start training.''',
        arguments='''
                epochs: set epochs for the training by default it is 10
                checkpoint_dir: directory from where the checkpoints should be loaded
                logdir: directory where the logs should be saved
                resume_path: The path/checkpoint from where the training should be resumed
                print_every: an argument to specify when the function should print or after how many epochs
                ''')
    def train(self,
              epochs: int = 10,
              checkpoint_dir: str = './checkpoints',
              logdir: str = './training_logs',
              resume_path: str = None,
              print_every: int = 10,
              **hyp) -> Tuple:
        for model_cfg, data_loader in zip(self.model_cfgs, self.data_loaders):
            train = get_train_pipeline(model_cfg['arch'])
            train(data_loader)
