import os
from typing import Dict

from utils.common_defs import class_header, method_header
from meghnad.repo.obj_det.yolov7.utils.datasets import create_dataloader as create_dataloader_yolov7
from meghnad.repo.obj_det.yolov5.utils.dataloaders import create_dataloader as create_dataloader_yolov5


@class_header(
    description='''
    Data loader for object detection.
    -----------------------------------
    Arguments
      data_path: Path to data root directory.
      data_cfg: Data config dictionary.
      model_cfg: Model config dictionary.
      augmentations: Augmentation dictionary.
    ''')
class PytorchObjDetDataLoader:
    def __init__(self,
                 data_path: str,
                 data_cfg: Dict,
                 model_cfg: Dict,
                 augmentations: Dict) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg

        self.train_loader, self.train_dataset = None, None
        self.val_loader, self.val_dataset = None, None
        self.test_loader, self.test_dataset = None, None

        self._load_datasets(data_path)

    @method_header(
        description='''
            Helper function for creating connecting dataset path to data directory.
            ''',
        arguments='''
            path : string : Local dataset path where data is located, it should be parent directory of path and is required to be a string.
            ''')
    def _config_connectors(
            self,
            path: str) -> None:

        self.connector = {}
        self.connector['trn_data_path'] = os.path.join(path, 'images')
        self.connector['trn_file_path'] = os.path.join(path, 'labels')
        self.connector['test_data_path'] = os.path.join(path, 'images')
        self.connector['test_file_path'] = os.path.join(path, 'labels')
        self.connector['val_data_path'] = os.path.join(path, 'images')
        self.connector['val_file_path'] = os.path.join(path, 'labels')

    def _load_datasets(self, data_path: str):
        datasets = ['train', 'val', 'test']
        for dataset in datasets:
            sub_data_path = os.path.join(data_path, dataset)
            loader, dataset = self._load(sub_data_path, dataset)
            setattr(self, f'{dataset}_loader', loader)
            setattr(self, f'{dataset}_dataset', dataset)

    def _load(self, path: str, dataset: str = 'train'):
        assert 'arch' in self.model_cfg
        rank = int(os.getenv('LOCAL_RANK', -1))
        if self.model_cfg['arch'] == 'yolov5':
            loaders = create_dataloader_yolov5(
                path,
                self.model_cfg['img_size'],
                8,
                32,
                False,
                hyp=self.model_cfg['hyp_params'],
                augment=True,
                cache=True,
                rect=False,
                rank=rank,
                workers=4,
                image_weights=False,
                quad=False,
                prefix='train: ',
                shuffle=True)
        elif self.model_cfg['arch'] == 'yolov7':
            class Opt:
                pass
            opt = Opt()
            opt.single_cls = False
            world_size = int(os.environ['WORLD_SIZE']
                             ) if 'WORLD_SIZE' in os.environ else 1
            loaders = create_dataloader_yolov7(
                path,
                self.model_cfg['img_size'],
                8,
                32,
                opt,
                hyp=self.model_cfg['hyp_params'],
                augment=True,
                cache=True,
                rect=False,
                rank=rank,
                world_size=world_size,
                workers=4,
                image_weights=False,
                quad=False,
                prefix='train: ')
        else:
            raise ValueError('Not supported arch')

        if dataset == 'train':
            return loaders
        return loaders, None
