import os
import sys
import unittest
# from meghnad.core.cv.obj_det.src.tf.trn import TFObjDetTrn
# from meghnad.core.cv.obj_det.src.tf.inference import TFObjDetPred
# from meghnad.core.cv.obj_det.src.pytorch.trn.trn import PyTorchObjDetTrn
# from meghnad.core.cv.obj_det.src.pytorch.inference.pred import PyTorchObjDetPred
import torch
torch.cuda.set_device(0)
from meghnad.core.cv.obj_det.src.trn import Trainer
from meghnad.core.cv.obj_det.src.inference.pred import Predictor
from utils.log import Log


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus, 'GPU')

log = Log()


def test_case1():
    settings = ['light']
    trainer = Trainer(settings)
    data_path = 'coco128.yaml'
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
        batch_size=4,
        epochs=10,
        hyp=hyp
    )

    print('Best mAP:', result.best_metric.map)
    print('Best path:', result.best_model_path)
    predictor = Predictor(result.best_model_path)
    val_path = './coco128/images/train2017'
    predictor.pred(val_path)


def _perform_tests():
    test_case1()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
