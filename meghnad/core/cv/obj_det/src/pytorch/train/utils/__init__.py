from typing import Dict


def get_train_pipeline(arch: str):
    if arch == 'yolov5':
        from meghnad.core.cv.obj_det.src.pytorch.train.utils.yolov5_train_utils import train as train_yolov5
        return train_yolov5
    elif arch == 'yolov7':
        from meghnad.core.cv.obj_det.src.pytorch.train.utils.yolov7_train_utils import train as train_yolov7
        return train_yolov7


class Opt:
    def __str__(self):
        repr_dict = dict()
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                repr_dict[k] = v
        return str(repr_dict)


def get_train_opt(model_cfg: Dict, **kwargs):
    opt = Opt()
    for k, v in model_cfg.items():
        setattr(opt, k, v)

    for k, v in kwargs.items():
        setattr(opt, k, v)
    return opt
