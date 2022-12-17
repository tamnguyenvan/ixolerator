from typing import Callable

from utils.common_defs import method_header


@method_header(
    description='''
    Get inference function.''',
    arguments='''
        arch: Model arch.''',
    returns='''
        Inference function.'''
)
def get_inference_pipeline(arch: str) -> Callable:
    if arch == 'yolov5':
        from meghnad.repo.obj_det.yolov5.val import run as test
        return test
    elif arch == 'yolov7':
        from meghnad.repo.obj_det.yolov7.test import test
        return test
