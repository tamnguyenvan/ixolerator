import unittest
from meghnad.core.cv.obj_det.src.tf.trn import TFObjDetTrn
from meghnad.core.cv.obj_det.src.tf.inference import TFObjDetPred
from meghnad.core.cv.obj_det.src.pytorch.trn.trn import PyTorchObjDetTrn
from meghnad.core.cv.obj_det.src.pytorch.inference.pred import PyTorchObjDetPred

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus, 'GPU')


def test_case1():
    """Training pipeline"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    success, best_model_path = trainer.trn(epochs=10)
    print('Best model path:', best_model_path)


def test_case2():
    """Test inference"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    success, best_model_path = trainer.trn(epochs=10)

    img_path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset\\images\\000a514fb1546570.jpg'
    predictor = TFObjDetPred(saved_dir=best_model_path)
    ret_value, (boxes, classes, scores) = predictor.pred(img_path)
    print(boxes.shape, classes.shape, scores.shape)


def test_case3():
    """Training + augmentations"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
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
    trainer.config_connectors(path, augmentations)
    trainer.trn(epochs=10)


def test_case4():
    """Test training pipeline + fine tune hyper parameters"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    trainer.trn(
        hyp={'optimizer': 'Adam', 'learning_rate': 1e-4, 'weight_decay': 1e-5}
    )


def test_case5():
    """Pytorch training pipeline test"""
    path = './coco128.yml'
    settings = ['light']
    trainer = PyTorchObjDetTrn(settings)
    trainer.config_connectors(path)
    trainer.trn(
        batch_size=1,
        epochs=10,
        imgsz=640,
        device='0'
    )


def test_case6():
    """Pytorch testing pipeline test"""
    path = './coco128.yml'
    settings = ['light']
    trainer = PyTorchObjDetTrn(settings)
    trainer.config_connectors(path)
    success, best_opt, best_path = trainer.trn(
        batch_size=1,
        epochs=1,
        imgsz=640,
        device='0',
        hyp={'fliplr': 0.4, 'lr0': 0.02,
             'lrf': 0.2, 'weight_decay': 0.0003,
             'translate': 0.2, 'scale': 0.8,
             'optimizer': 'Adam'}
    )

    print('=' * 50)
    print('====== Path to the best model:', best_path)
    print('=' * 50)

    tester = PyTorchObjDetPred(best_opt, best_path)
    img_path = './coco128/images/train2017/000000000009.jpg'
    tester.pred(img_path)


def _perform_tests():
    test_case6()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
