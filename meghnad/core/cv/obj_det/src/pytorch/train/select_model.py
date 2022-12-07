from typing import Dict

from meghnad.core.cv.obj_det.src.tensorflow.model_loader import ssd
from utils.common_defs import class_header

from meghnad.repo.obj_det.yolov7.models.yolo import Model as YOLOv7
from meghnad.repo.obj_det.yolov5.models.yolo import Model as YOLOv5

__all__ = ['PytorchObjDetSelectModel']


@class_header(
    description='''
    Select Model and setup configurations''')
class PytorchObjDetSelectModel:
    def __init__(self, model_configs: Dict):
        self.best_model = None

        self.model_configs = model_configs
        self.models = []
        for model_config in self.model_configs:
            if model_config['arch'] == 'yolov5':
                model = YOLOv5(model_config)
            elif model_config['arch'] == 'yolov7':
                model = YOLOv7(model_config)
            else:
                raise ValueError('Not supported arch')
            self.models.append(model)
