from typing import Optional, Tuple, Any
import time
from pathlib import Path

import cv2
import torch
from numpy import random

from meghnad.repo.obj_det.yolov7.models.experimental import attempt_load
from meghnad.repo.obj_det.yolov7.utils.datasets import LoadImages
from meghnad.repo.obj_det.yolov7.utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging, increment_path
from meghnad.repo.obj_det.yolov7.utils.plots import plot_one_box
from meghnad.repo.obj_det.yolov7.utils.torch_utils import select_device, time_synchronized

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header


__all__ = ['PyTorchObjDetPred']

log = Log()


class Opt:
    pass


@class_header(
    description='''
    Class for Object detection predictions''')
class PyTorchObjDetPred:
    def __init__(self,
                 weights: str,
                 output_dir: Optional[str] = './results') -> None:
        self.weights = weights
        self.output_dir = output_dir

    def pred(self,
                input: Any,
                conf_thres: float = 0.25,
                iou_thres: float = 0.45) -> Tuple:
        opt = Opt()
        opt.nosave = False
        opt.conf_thres = conf_thres
        opt.iou_thres = iou_thres
        opt.project = 'runs/test'
        opt.name = 'exp'
        opt.device = ''
        opt.classes = None
        opt.exist_ok = False
        opt.no_trace = True
        opt.augment = False
        opt.agnostic_nms = False

        weights = self.weights

        save_img = not opt.nosave and not input.endswith(
            '.txt')  # save inference images
        webcam = input.isnumeric() or input.endswith('.txt') or input.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name,
                        exist_ok=opt.exist_ok))  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride

        imgsz = torch.load(weights, map_location=device)['img_size']
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            model.half()  # to FP16
        dataset = LoadImages(input, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                    ), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(
                        dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        if save_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label,
                                         color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(
                    f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(
                            f" The image with the result is saved in: {save_path}")

        print(f'Done. ({time.time() - t0:.3f}s)')
