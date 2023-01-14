import os
import sys
import tempfile
import json
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header
from meghnad.core.cv.obj_det.src.backend.tf.inference.pred_utils import draw_bboxes
from meghnad.core.cv.obj_det.src.backend.tf.model_loader.utils import decode, compute_nms
from meghnad.core.cv.obj_det.src.backend.tf.data_loader import TFObjDetDataLoader
from meghnad.core.cv.obj_det.src.utils.general import get_sync_dir

log = Log()

__all__ = ['TFObjDetEval']


@class_header(
    description='''
    Evaluation class to evaluate models after training''')
class TFObjDetEval:
    def __init__(self, model: tf.keras.Model):
        self.model = model

    @method_header(
        description='''
                ''',
        arguments='''
                data_loader: data_loader to load data
                phase [:optional]: select which data to be loaded by default it is (validation)
                class_map: map with score and class_labels
                nms_threshold: NMS threshold,
                max_predictions: number of predictions per image,
                image_out_dir: directory to which image should be placed after eval
                draw_predictions: if true draw prediction on the images also
                ''',
        returns='''
                a 2 value pair map, map50 containing evaluation stats''')
    def eval(self,
             data_loader: TFObjDetDataLoader,
             phase: str = 'validation',
             class_map: Dict = dict(),
             score_threshold: float = 0.4,
             nms_threshold: float = 0.5,
             max_predictions: int = 100,
             image_out_dir: str = 'results',
             draw_predictions: bool = False) -> Tuple[float]:
        if self.model is None:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is not fitted yet")
            return ret_values.IXO_RET_INVALID_INPUTS

        sync_dir = get_sync_dir()
        image_out_dir = os.path.join(sync_dir, image_out_dir)
        results = {'annotations': []}
        ann_id = 0
        if phase == 'validation':
            dataset = data_loader.validation_dataset
        else:
            dataset = data_loader.test_dataset

        for batch_image_ids, batch_image_shapes, batch_images, _, _ in dataset:
            batch_confs, batch_locs = self.model(
                batch_images, training=False)
            for image, image_id, image_shape, confs, locs in zip(
                batch_images, batch_image_ids, batch_image_shapes, batch_confs, batch_locs
            ):

                image_id = int(image_id)
                image_id_str = str(image_id)
                image_height, image_width = image_shape.numpy()[:2]

                confs = tf.math.softmax(confs, axis=-1)
                classes = tf.math.argmax(confs, axis=-1)
                scores = tf.math.reduce_max(confs, axis=-1)

                boxes = decode(data_loader.default_boxes, locs)

                out_boxes = []
                out_labels = []
                out_scores = []

                for c in range(1, data_loader.num_classes):
                    cls_scores = confs[:, c]

                    score_idx = cls_scores > score_threshold
                    cls_boxes = boxes[score_idx]
                    cls_scores = cls_scores[score_idx]

                    nms_idx = compute_nms(
                        cls_boxes, cls_scores, nms_threshold, max_predictions)
                    cls_boxes = tf.gather(cls_boxes, nms_idx)
                    cls_scores = tf.gather(cls_scores, nms_idx)
                    cls_labels = [c] * cls_boxes.shape[0]

                    out_boxes.append(cls_boxes)
                    out_labels.extend(cls_labels)
                    out_scores.append(cls_scores)
                out_boxes = tf.concat(out_boxes, axis=0)
                out_scores = tf.concat(out_scores, axis=0)

                boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
                boxes_resized = boxes * \
                    np.array([[*data_loader.input_shape * 2]]
                             ).astype(np.float32)
                boxes_resized = boxes_resized.astype(np.int32).tolist()
                boxes = boxes * \
                    np.array([[image_width, image_height, image_width, image_height]]).astype(
                        np.float32)
                boxes = boxes.astype(np.int32).tolist()
                classes = np.array(out_labels)
                scores = out_scores.numpy()
                if draw_predictions:
                    dest_path = os.path.join(
                        image_out_dir, image_id_str + '.jpg')
                    image *= 255
                    pred_bbox_image = draw_bboxes(
                        image[..., ::-1].numpy(), boxes_resized, classes, scores, class_map)
                    cv2.imwrite(dest_path, pred_bbox_image)

                for box, cls, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = box
                    ann_id += 1
                    results['annotations'].append({
                        'id': ann_id,
                        'image_id': image_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': (x2 - x1) * (y2 - y1),
                        'category_id': int(cls),
                        'score': float(score)
                    })

        ann_json = None
        if phase == 'validation':
            ann_json = data_loader.val_ann_file
        elif phase == 'test':
            ann_json = data_loader.test_ann_file
        else:
            raise FileNotFoundError(
                'Not found ground truth annotation file')

        pred_file = tempfile.NamedTemporaryFile('wt').name
        with open(pred_file, 'wt') as f:
            json.dump(results, f)

        gt = COCO(ann_json)  # init annotations api
        pred = COCO(pred_file)
        eval = COCOeval(gt, pred, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]
        if os.path.isfile(pred_file):
            os.remove(pred_file)
        return map, map50
