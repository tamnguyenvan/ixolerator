import sys
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
from meghnad.core.cv.obj_det.src.utils.general import get_sync_dir

from utils.log import Log
from utils.common_defs import class_header, method_header


__all__ = ['PyTorchObjDetPred']

log = Log()


@class_header(
    description='''
    Class for Object detection predictions''')
class PyTorchObjDetPred:
    def __init__(self,
                 weights: str,
                 output_dir: Optional[str] = './results') -> None:
        self.weights = weights
        self.output_dir = output_dir

        # Initialize
        set_logging()
        self.device = select_device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(
            weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride

        # Check image size
        self.imgsz = torch.load(weights, map_location=self.device)['img_size']
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once

    @method_header(
        description="""Curates directories, runs inference, performs post processing and processes detections
        """,
        arguments="""
            input: image
            conf_thres: Minimum confidence required for the object to be shown as detection
            iou_thres: Intersection over Union Threshold
        """,
        returns="""Prints out the time taken for actual inferencing, and then the post processing steps""")
    def pred(self,
             input: Any,
             conf_thres: float = 0.25,
             iou_thres: float = 0.45,
             max_predictions: int = 100,
             save_img: bool = True) -> Tuple:
        device = self.device
        half = self.half
        model = self.model
        imgsz = self.imgsz

        # Load dataset
        dataset = LoadImages(input, img_size=self.imgsz, stride=self.stride)

        webcam = input.isnumeric() or input.endswith('.txt') or input.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        project = 'runs/test'
        sync_dir = Path(get_sync_dir())
        # save_dir = sync_dir / project / opt.name
        save_dir = sync_dir / self.output_dir  # TODO
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
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
                    model(img, augment=False)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)
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
                log.STATUS(sys._getframe().f_lineno,
                           __file__, __name__,
                           f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        log.STATUS(sys._getframe().f_lineno,
                                   __file__, __name__,
                                   f" The image with the result is saved in: {save_path}")

        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   f'Done. ({time.time() - t0:.3f}s)')
