
def get_inference_pipeline(arch: str):
    if arch == 'yolov5':
        from meghnad.repo.obj_det.yolov5.val import run as test
        return test
    elif arch == 'yolov7':
        from meghnad.repo.obj_det.yolov7.test import test
        return test
