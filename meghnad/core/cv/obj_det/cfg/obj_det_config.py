#######################################################################################################################
# Configurations for Object Detection.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Prudhvi Raju
#######################################################################################################################
from typing import List
from dataclasses import dataclass
from meghnad.cfg.config import MeghnadConfig

__all__ = ['ObjDetConfig', 'BACKENDS']


@dataclass
class BACKENDS:
    TENSORFLOW: str = 'tensorflow'
    PYTORCH: str = 'pytorch'


_obj_det_cfg = {
    'models':
    {
        'MobileNetV2',
        'EfficientNetB3',
        'EfficientNetB4',
        'EfficientNetB5',
        'EfficientNetV2S',
        'EfficientNetV2M',
        'EfficientNetV2L',
        'YOLOv5',
        'YOLOv7'
    },
    'model_cfg':
    {
        'MobileNetV2':
        {
            'config_name': 'MobileNetV2',
            'arch': 'MobileNetV2',
            'backend': 'tensorflow',
            'pretrained': None,
            'input_shape': (300, 300, 3),
            'num_classes': 10 + 1,  # num_classes + background
            'classes': [],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 6, 4, 4],
            'feature_map_sizes': [19, 10, 5, 3, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.05],
            'neg_ratio': 3,
            'augmentations': {
                'train':
                {
                    'resize': {'width': 300, 'height': 300},
                    'random_fliplr': {'p': 0.5},
                    'random_brightness': {'p': 0.2},
                    'normalize': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
                },
                'test':
                {
                    'resize': {'width': 300, 'height': 300},
                    'normalize': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
                }
            },
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'Adam',
                'learning_rate': 0.0001,
                'weight_decay': 5e-4
            }
        },
        'EfficientNetB3':
        {
            'config_name': 'EfficientNetB3',
            'arch': 'EfficientNetB3',
            'backend': 'tensorflow',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
                'weight_decay': 5e-4
            }
        },
        'EfficientNetB4': {
            'config_name': 'EfficientNetB4',
            'arch': 'EfficientNetB4',
            'backend': 'tensorflow',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
                'weight_decay': 5e-4
            }
        },
        'EfficientNetB5': {
            'config_name': 'EfficientNetB5',
            'arch': 'EfficientNetB5',
            'backend': 'tensorflow',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
                'weight_decay': 5e-4
            }
        },
        'EfficientNetV2S': {
            'config_name': 'EfficientNetV2S',
            'arch': 'EfficientNetV2S',
            'backend': 'tensorflow',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
                'weight_decay': 5e-4
            }
        },
        'EfficientNetV2M': {
            'config_name': 'EfficientNetV2M',
            'arch': 'EfficientNetV2M',
            'backend': 'tensorflow',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
                'weight_decay': 5e-4
            }
        },
        'EfficientNetV2L': {
            'config_name': 'EfficientNetV2L',
            'arch': 'EfficientNetV2L',
            'backend': 'tensorflow',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
                'weight_decay': 5e-4
            }
        },
        'YOLOv5': {
            'config_name': 'YOLOv5',
            'arch': 'yolov5',
            'backend': 'pytorch',
            'weights': 'yolov5s.pt',
            'imgsz': 640,
            'cfg': '',
            'hyp': 'data/hyps/hyp.scratch-med.yaml',
            'rect': False,
            'resume': False,
            'nosave': False,
            'noval': False,
            'noautoanchor': False,
            'noplots': False,
            'evolve': False,
            'cache': '',
            'image_weights': False,
            'multi_scale': False,
            'single_cls': False,
            'optimizer': 'SGD',
            'sync_bn': False,
            'workers': 8,
            'project': 'runs/train',
            'name': 'yolov5s',
            'exist_ok': False,
            'quad': False,
            'cos_lr': False,
            'label_smoothing': 0.0,
            'patience': 100,
            'freeze': [0],
            'save_period': -1,
            'seed': 0,
        },
        'YOLOv7Light': {
            'config_name': 'YOLOv7Light',
            'arch': 'yolov7',
            'backend': 'pytorch',
            'weights': 'yolov7.pt',
            'img_size': [640],
            'cfg': '',
            'hyp': 'data/hyp.scratch.p5.yaml',
            'rect': False,
            'resume': False,
            'nosave': False,
            'noautoanchor': False,
            'notest': False,
            'evolve': False,
            'bucket': '',
            'cache_images': False,
            'image_weights': False,
            'multi_scale': False,
            'single_cls': False,
            'adam': False,
            'sync_bn': False,
            'workers': 4,
            'project': 'runs/train',
            'name': 'yolov7_light',
            'exist_ok': False,
            'quad': False,
            'linear_lr': False,
            'label_smoothing': 0.0,
            'freeze': [0],
            'save_period': -1,
            'local_rank': -1,
            'v5_metric': False
        },
        'YOLOv7Large': {
            'config_name': 'YOLOv7Large',
            'arch': 'yolov7',
            'backend': 'pytorch',
            'weights': 'yolov7.pt',
            'img_size': [1280],
            'cfg': '',
            'hyp': 'data/hyp.scratch.p5.yaml',
            'rect': False,
            'resume': False,
            'nosave': False,
            'noautoanchor': False,
            'notest': False,
            'evolve': False,
            'bucket': '',
            'cache_images': False,
            'image_weights': False,
            'multi_scale': False,
            'single_cls': False,
            'adam': False,
            'sync_bn': False,
            'workers': 4,
            'project': 'runs/train',
            'name': 'yolov7_large',
            'exist_ok': False,
            'quad': False,
            'linear_lr': False,
            'label_smoothing': 0.0,
            'freeze': [0],
            'save_period': -1,
            'local_rank': -1,
            'v5_metric': False
        }
    },
    'model_settings':
    {
        'default_models': ['MobileNetV2', 'EfficientNetB3'],
        'light_models': ['YOLOv5', 'MobileNetV2'],
        'large_models': ['YOLOv7Large']
    }
}


class ObjDetConfig(MeghnadConfig):
    def __init__(self, *args):
        super().__init__()

    def get_model_cfg(self, model_name: str) -> dict:
        try:
            return _obj_det_cfg['model_cfg'].copy()[model_name]
        except:
            return _obj_det_cfg['model_cfg'][model_name]

    def get_data_cfg(self) -> dict:
        try:
            return _obj_det_cfg['data_cfg'].copy()
        except:
            return _obj_det_cfg['data_cfg']

    def get_model_settings(self, setting_name: str = None) -> dict:
        if setting_name and setting_name in _obj_det_cfg['model_settings']:
            try:
                return _obj_det_cfg['model_settings'][setting_name].copy()
            except:
                return _obj_det_cfg['model_settings'][setting_name]

    def set_user_cfg(self, user_cfg):
        for key in user_cfg:
            self.user_cfg[key] = user_cfg[key]

    def get_user_cfg(self) -> dict:
        return self.user_cfg

    def get_models_by_names(self) -> List[str]:
        models = []
        for key in _obj_det_cfg['model_archs']:
            models.append(key)
        return models
