import os
import sys
from typing import Optional, Tuple, Any

from utils.log import Log
from utils import ret_values
from utils.common_defs import class_header, method_header
from meghnad.core.cv.obj_det.src.pt.inference.pred import PTObjDetPred
from meghnad.core.cv.obj_det.src.tf.inference.pred import TFObjDetPred

log = Log()

@class_header(
    description='''
    Wrapper class for Object detection predictions''')
class Predictor:
    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path
        self.predictor = self._load(ckpt_path)

    @method_header(
        description="""Method that applies Pytorch or Tensorflow pred classes based on the backend selected
            """,
        arguments="""
                path: path to weights file
                """,
        returns="""PT or TF predictor class""")
    def _load(self, path: str):
        if path is None:
            log.ERROR(sys._getframe().f_lineno,
                        __file__, __name__,
                        f'Unable to load model from {path}')
            return ret_values.IXO_RET_INVALID_INPUTS
            #raise Exception(f'Unable to load model from {path}')
        if os.path.isfile(path):
            # pytorch weights
            return PTObjDetPred(path)
        elif os.path.isdir(path):
            # tensorflow saved model
            return TFObjDetPred(path)
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      f'Unable to load model from {path}')
            return ret_values.IXO_RET_INVALID_INPUTS

            #raise Exception(f'Unable to load model from {path}')

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
