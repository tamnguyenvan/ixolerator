import os
from typing import List, Tuple, Dict

import numpy as np

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header

from meghnad.core.cv.obj_det.src.backend.pytorch.trn.trn_utils import get_train_pipeline, get_train_opt
from meghnad.core.cv.obj_det.src.trn.metric import Metric
from connectors.aws.s3.config import S3Config


__all__ = ['PyTorchObjDetTrn']


log = Log()


@method_header(
    description="""Combines precision, recall, mAP@0.5, and mAP@0.5:0.95 to final metric.
    """,
    arguments="""
        x: A 2D-array of metrics.
    """,
    returns="""Final metric.""")
def fitness(x: np.ndarray) -> float:
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


@class_header(
    description='''
        Class for object detection model training''')
class PyTorchObjDetTrn:
    def __init__(self, model_cfgs: List[Dict]) -> None:
        self.model_cfgs = model_cfgs
        self.data_path = None
        self.best_model_path = None

    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data (should point to the file in case of a single file, should point to
                the directory in case data exists in multiple files in a directory structure)
                ''')
    def config_connectors(self, data_path: str) -> None:
        self.data_path = data_path

    @method_header(
        description='''
                Function to set training configurations and start training.''',
        arguments='''
                epochs: number of complete passes through the training dataset. Set epochs for the training by default as 10
                imgsz: image size
                batch_size: number of samples processed before the model is updated
                workers: the number of sub process that ingest data
                hyp: tuple that accepts the hyper parameters from data/hyps/
                ''',
        returns='''
                Best model path for prediction'''
    )
    def trn(self,
            batch_size: int = 16,
            epochs: int = 10,
            imgsz: int = 640,
            device: str = '',
            workers: int = 8,
            hyp: Dict = None) -> Tuple:
        best_fitness = 0.0
        best_path = None
        for model_cfg in self.model_cfgs:
            train_pipeline = get_train_pipeline(model_cfg['arch'])
            opt = get_train_opt(
                model_cfg,
                data=self.data_path,
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                device=device,
                workers=workers,
                hyp=hyp
            )

            results, best = train_pipeline(opt)
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
                best_path = best

        metric = Metric(map=best_fitness)
        return ret_values.IXO_RET_SUCCESS, metric, best_path
