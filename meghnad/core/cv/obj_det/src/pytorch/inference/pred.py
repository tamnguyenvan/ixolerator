from pathlib import Path
from typing import Optional, Tuple, Any
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from meghnad.repo.obj_det.yolov7.models.experimental import attempt_load
from meghnad.repo.obj_det.yolov7.utils.datasets import LoadStreams, LoadImages
from meghnad.repo.obj_det.yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from meghnad.repo.obj_det.yolov7.utils.plots import plot_one_box
from meghnad.repo.obj_det.yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header


__all__ = ['PytorchObjDetPred']

log = Log()


class Opt:
    pass


@class_header(
    description='''
    Class for Object detection predictions''')
class PytorchObjDetPred:
    def __init__(self,
                 weights: str,
                 output_dir: Optional[str] = './results') -> None:
        self.weights = weights
        self.output_dir = output_dir

    def predict(self,
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
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
        #                                                       exist_ok=True)  # make dir
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

        # if trace:
        #     model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()  # to FP16

        # # Second-stage classifier
        # classify = False
        # if classify:
        #     modelc = load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load(
        #         'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        # vid_path, vid_writer = None, None
        # if webcam:
        #     view_img = check_imshow()
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # else:
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

            # # Apply Classifier
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

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
                txt_path = str(save_dir / 'labels' / p.stem) + \
                    ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
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
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                        #                       ) / gn).view(-1).tolist()  # normalized xywh
                        #     # label format
                        #     line = (
                        #         cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() %
                        #                 line + '\n')

                        if save_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label,
                                         color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(
                    f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # # Stream results
                # if view_img:
                #     cv2.imshow(str(p), im0)
                #     cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(
                            f" The image with the result is saved in: {save_path}")
                    # else:  # 'video' or 'stream'
                    #     if vid_path != save_path:  # new video
                    #         vid_path = save_path
                    #         if isinstance(vid_writer, cv2.VideoWriter):
                    #             vid_writer.release()  # release previous video writer
                    #         if vid_cap:  # video
                    #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    #         else:  # stream
                    #             fps, w, h = 30, im0.shape[1], im0.shape[0]
                    #             save_path += '.mp4'
                    #         vid_writer = cv2.VideoWriter(
                    #             save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    #     vid_writer.write(im0)

        # if save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
