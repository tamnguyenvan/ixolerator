from typing import Dict
import importlib


def build_loader(cfg: Dict, **kwargs):
    assert cfg['arch'], 'Not found model arch'
    arch = cfg['arch']
    module_name = f'{__name__}.{arch}_data_loader'
    module = importlib.import_module(module_name)
    create_dataloader_fn = module.__dict__['create_dataloader']
    return create_dataloader_fn(**kwargs)
