import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

from utils import ret_values
from meghnad.core.cv.obj_det.cfg import ObjDetConfig, BACKENDS
from meghnad.core.cv.obj_det.src.backend.tf.trn import TFObjDetTrn
from meghnad.core.cv.obj_det.src.backend.pytorch.trn import PyTorchObjDetTrn
from meghnad.core.cv.obj_det.src.trn.metric import Metric
from utils.common_defs import class_header, method_header


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
def load_config_from_settings(settings: List[str]) -> List:
    settings = [f'{setting}_models' for setting in settings]
    cfg_obj = ObjDetConfig()

    model_cfgs = []
    for setting in settings:
        model_names = cfg_obj.get_model_settings(setting)
        for model_name in model_names:
            model_cfg = cfg_obj.get_model_cfg(model_name)
            model_cfgs.append(model_cfg)
    return model_cfgs


@dataclass
class TrainingResult:
    best_metric: Metric = None
    best_model_path: str = None


class Trainer:
    def __init__(self, settings: List[str]) -> None:
        self.settings = settings
        self.model_cfgs = load_config_from_settings(settings)
        self.data_pat = None

        tf_model_cfgs = []
        pt_model_cfgs = []
        for model_cfg in self.model_cfgs:
            backend = model_cfg['backend']
            if backend == BACKENDS.TENSORFLOW:
                tf_model_cfgs.append(model_cfg)
            elif backend == BACKENDS.PYTORCH:
                pt_model_cfgs.append(model_cfg)
            else:
                raise ValueError(f'Unsupported backend: {backend}')
        self.tf_trainer = TFObjDetTrn(tf_model_cfgs)
        self.pt_trainer = PyTorchObjDetTrn(pt_model_cfgs)

    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data (should point to the file in case of a single file, should point to
                the directory in case data exists in multiple files in a directory structure)
                ''')
    def config_connectors(self, data_path: str, augmentations: Dict = None) -> None:
        self.tf_trainer.config_connectors(data_path, augmentations)
        self.pt_trainer.config_connectors(data_path)

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
    def trn(self,
            batch_size: int = 32,
            epochs: int = 5) -> Tuple:
        # success, pt_best_metric, pt_best_path = self.pt_trainer.trn(
        #     batch_size=batch_size,
        #     epochs=epochs
        # )
        success, tf_best_metric, tf_best_path = self.tf_trainer.trn(
            batch_size=batch_size,
            epochs=epochs
        )

        # compare
        best_metric = None
        # if pt_best_metric.map > tf_best_metric.map:
        #     best_metric = pt_best_metric
        #     best_model_path = pt_best_path
        # else:
        best_metric = tf_best_metric
        best_model_path = tf_best_path

        result = TrainingResult(
            best_metric=best_metric,
            best_model_path=best_model_path
        )
        return ret_values.IXO_RET_SUCCESS, result
