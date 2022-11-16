import os
import sys
import tempfile
import json

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

from utils import ret_values
from utils.log import Log
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.inference.vis_utils import draw_bboxes
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.anchors import generate_default_boxes
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.utils.box_utils import decode, compute_nms

log = Log()


class ModelEvaluator:
    def __init__(self,
                 model_loader=None,
                 model_config=None,
                 data_loader=None,
                 weights=None,
                 phase='test',
                 image_out_dir = '',
                 draw_predictions = False,
                 from_hub = False):
        self.model_loader = model_loader
        self.data_loader = data_loader
        self.weights = weights
        self.dataset = data_loader.validation_dataset if phase == 'validation' else data_loader.test_dataset
        self.phase = phase
        self.image_out_dir = image_out_dir
        self.draw_predictions = draw_predictions
        self.default_boxes = generate_default_boxes(
            model_config['scales'],
            model_config['feature_map_sizes'],
            model_config['aspect_ratios']
        )

        if weights and not from_hub:
            self.model_loader.model.load_weights(weights)
            print(f'Loaded pretrained weights from {weights}')

    def eval(self):
        if self.model_loader.model is None:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is not fitted yet")
            return ret_values.IXO_RET_INVALID_INPUTS
        # if len(self.dataset.cardinality().shape) == 0:
        #     log.ERROR(sys._getframe().f_lineno,
        #               __file__, __name__, "Test data is empty")
        #     return ret_values.IXO_RET_INVALID_INPUTS

        results = {'annotations': []}
        ann_id = 0
        for batch_image_ids, batch_image_shapes, batch_images, _, _ in self.dataset:
            batch_confs, batch_locs = self.model_loader.model(
                tf.cast(batch_images, tf.uint8)),[0]#, training=False)
            for image, image_id, image_shape, confs, locs in zip(
                batch_images, batch_image_ids, batch_image_shapes, [batch_confs], batch_locs
            ):
                image_id = int(image_id)
                image_id_str = str(image_id)
                image_height, image_width = image_shape.numpy()[:2]
                #print(confs["detection_boxes"].numpy().shape)

                boxes = confs["detection_boxes"]

                #boxes_resized = boxes
                classes = confs["detection_classes"].numpy()[0]
                scores = confs["detection_scores"].numpy()[0]


                print(classes.shape, scores.shape)
##                confs = tf.math.softmax(confs, axis=-1)
##                classes = tf.math.argmax(confs, axis=-1)
##                scores = tf.math.reduce_max(confs, axis=-1)
##
##                boxes = decode(self.default_boxes, locs)
##
##                out_boxes = []
##                out_labels = []
##                out_scores = []

##                for c in range(1, 3):
##                    cls_scores = confs[:, c]
##
##                    score_idx = cls_scores > 0.6
##                    # cls_boxes = tf.boolean_mask(boxes, score_idx)
##                    # cls_scores = tf.boolean_mask(cls_scores, score_idx)
##                    cls_boxes = boxes[score_idx]
##                    cls_scores = cls_scores[score_idx]
##
##                    nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
##                    cls_boxes = tf.gather(cls_boxes, nms_idx)
##                    cls_scores = tf.gather(cls_scores, nms_idx)
##                    cls_labels = [c] * cls_boxes.shape[0]
##
##                    out_boxes.append(cls_boxes)
##                    out_labels.extend(cls_labels)
##                    out_scores.append(cls_scores)
##
##                out_boxes = tf.concat(out_boxes, axis=0)
##                out_scores = tf.concat(out_scores, axis=0)
##
                boxes = tf.clip_by_value(boxes, 0.0, 1.0).numpy()[0]
                filter = scores > 0.6
                boxes = boxes[filter]
                classes = classes[filter]
                scores = scores[filter]

                print("shape", boxes.shape,classes.shape,scores.shape)
                boxes_resized = boxes * np.array([[*self.data_loader.img_size*2]]).astype(np.float32)
                boxes_resized = boxes_resized.astype(np.int32).tolist()
                boxes = boxes * np.array([[image_width, image_height, image_width, image_height]]).astype(np.float32)
                boxes = boxes.astype(np.int32).tolist()
##                classes = np.array(out_labels)
##                scores = out_scores.numpy()

                if (self.draw_predictions):
                    #print(image[::-1])
                    dest_path = os.path.join(self.image_out_dir, image_id_str+'.jpg')
                    #image *= 255
                    #image = tf.reshape(image, (image_height, image_width, 3))
                    pred_bbox_image = draw_bboxes(image[...,::-1].numpy(), boxes_resized, classes, scores)
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

        print(results)
        ann_json = None
        if self.phase == 'validation':
            ann_json = self.data_loader.connector['val_file_path']
        elif self.phase == 'test':
            ann_json = self.data_loader.connector['test_file_path']
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

        # return ret_values.IXO_RET_SUCCESS, loss, metrics