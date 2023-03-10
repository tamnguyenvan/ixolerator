import sys, os
import unittest

from meghnad.core.cv.obj_det.src.trn_wrapper import Trainer
from meghnad.core.cv.obj_det.src.pred_wrapper import Predictor
from utils.log import Log

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_of_gpus = torch.cuda.device_count()
torch.cuda.set_device(0)

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
   tf.config.set_visible_devices(gpus, 'GPU')

log = Log()


def test_case1():
    settings = ['light']
    trainer = Trainer(settings)
    data_path = 'data/coco128.yml'
    augmentations = {
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
    }
    hyp = {'fliplr': 0.4, 'lr0': 0.02,
           'lrf': 0.2, 'weight_decay': 0.0003,
           'translate': 0.2, 'scale': 0.8,
           'optimizer': 'Adam'}

    trainer.config_connectors(data_path, augmentations=augmentations)
    _, result = trainer.trn(
        batch_size=2,
        epochs=5,
        workers=4,
        device='0',
        hyp=hyp
    )

    # log.VERBOSE(sys._getframe().f_lineno,
    #            __file__, __name__,
    #            f'Best mAP: {result.best_metric.map}')
    #
    # log.VERBOSE(sys._getframe().f_lineno,
    #            __file__, __name__,
    #            f'Best path: {result.best_model_path}')

    print('Best mAP:', result.best_metric.map)
    print('Best path:', result.best_model_path)
    predictor = Predictor(result.best_model_path)
    val_path = 'D:/ixo_data/data/coco128/images/train2017'
    predictor.pred(val_path)


def test_case2():
    """Test Zero-Shot Object Detection model"""
    settings = ['zero_shot']
    trainer = Trainer(settings)
    data_path = 'data/coco128.yml'
    augmentations = {
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
    }
    hyp = {'fliplr': 0.4, 'lr0': 0.02,
           'lrf': 0.2, 'weight_decay': 0.0003,
           'translate': 0.2, 'scale': 0.8,
           'optimizer': 'Adam'}

    trainer.config_connectors(data_path, augmentations=augmentations)
    _, result = trainer.trn(
        batch_size=2,
        epochs=5,
        workers=4,
        device='0',
        hyp=hyp
    )

    # log.VERBOSE(sys._getframe().f_lineno,
    #            __file__, __name__,
    #            f'Best mAP: {result.best_metric.map}')
    #
    # log.VERBOSE(sys._getframe().f_lineno,
    #            __file__, __name__,
    #            f'Best path: {result.best_model_path}')

    print('Best mAP:', result.best_metric.map)
    print('Best path:', result.best_model_path)
    predictor = Predictor(result.best_model_path)
    val_path = 'D:/ixo_data/data/coco128/images/train2017'
    predictor.pred(val_path)


def _perform_tests():
    test_case1()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
