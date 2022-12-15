import unittest
from meghnad.core.cv.obj_det.src.tensorflow.train import TFObjDetTrn
from meghnad.core.cv.obj_det.src.tensorflow.inference import TFObjDetPred
from meghnad.core.cv.obj_det.src.pytorch.train.train import PytorchObjDetTrn
from meghnad.core.cv.obj_det.src.pytorch.inference.pred import PytorchObjDetPred
from meghnad.core.cv.obj_det.src.pytorch.data_loader import build_loader
from meghnad.core.cv.obj_det.cfg import ObjDetConfig


def test_case1():
    """Training pipeline"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    success, best_model_path = trainer.train(epochs=10)
    print('Best model path:', best_model_path)


def test_case2():
    """Test inference"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    success, best_model_path = trainer.train(epochs=10)

    img_path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset\\images\\000a514fb1546570.jpg'
    predictor = TFObjDetPred(saved_dir=best_model_path)
    ret_value, (boxes, classes, scores) = predictor.predict(img_path)
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
    trainer.train(epochs=10)


def test_case4():
    """Test training pipeline + fine tune hypyerparameters"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    trainer.train(
        hyp={'optimizer': 'Adam', 'learning_rate': 1e-4}
    )


def test_case5():
    """Pytorch YOLOv5/v7 data loaders (internal test)"""
    path = './coco128'
    cfg = ObjDetConfig()
    model_cfg = cfg.get_model_cfg('YOLOv7')

    data_loader = build_loader(
        model_cfg, path=path, imgsz=640, batch_size=4, stride=32)


def test_case6():
    """Pytorch training pipeline test"""
    path = './coco128.yml'
    settings = ['light']
    trainer = PytorchObjDetTrn(settings)
    trainer.config_connectors(path)
    trainer.train(
        batch_size=1,
        epochs=10,
        imgsz=640
    )


def test_case7():
    """Pytorch testing pipeline test"""
    path = './coco128.yml'
    settings = ['light']
    trainer = PytorchObjDetTrn(settings)
    trainer.config_connectors(path)
    best_path = trainer.train(
        batch_size=1,
        epochs=2,
        imgsz=640
    )

    tester = PytorchObjDetPred(best_path)
    img_path = './coco128/images/train2017/000000000009.jpg'
    tester.predict(img_path)


def _perform_tests():
    test_case7()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
