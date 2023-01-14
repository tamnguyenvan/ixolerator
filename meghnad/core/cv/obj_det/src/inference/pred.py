import os
from typing import Optional, Tuple, Any

from utils.common_defs import class_header, method_header
from meghnad.core.cv.obj_det.src.backend.pytorch.inference.pred import PyTorchObjDetPred
from meghnad.core.cv.obj_det.src.backend.tf.inference.pred import TFObjDetPred


class Predictor:
    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path
        self.predictor = self._load(ckpt_path)

    def _load(self, path: str):
        if path is None:
            raise Exception(f'Unable to load model from {path}')
        if os.path.isdir(path):
            # pytorch weights
            return PyTorchObjDetPred(path)
        elif os.path.isfile(path):
            # tensorflow saved model
            return TFObjDetPred(path)
        else:
            raise Exception(f'Unable to load model from {path}')

    @method_header(
        description="""Curates directories, runs inference, performs post processing and processes detections
        """,
        arguments="""
            input: image
            conf_thres: Minimum confidence required for the object to be shown as detection
            iou_thres: Intersection over Union Threshold
        """,
        returns="""Prints out the time taken for actual inferencing, and then the post processing steps""")
    def pred(self,
             input: Any,
             conf_thres: float = 0.25,
             iou_thres: float = 0.45,
             max_predictions: int = 100,
             save_img: bool = True) -> Tuple:
        return self.predictor.pred(
            input, conf_thres, iou_thres, max_predictions, save_img)
