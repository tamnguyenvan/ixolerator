import itertools
import math
import tensorflow as tf


def generate_default_boxes(scales, feature_map_sizes, aspect_ratios):
    """ Generate default boxes for all feature maps
    Args:
        config: information of feature maps
            scales: boxes' size relative to image's size
            feature_map_sizes: sizes of feature maps
            aspect_ratios: box aspect_ratios used in each feature maps
    Returns:
        default_boxes: tensor of shape (num_default, 4)
                       with format (cx, cy, w, h)
    """
    print(scales, len(scales))
    print(feature_map_sizes, len(feature_map_sizes))
    print(aspect_ratios, len(aspect_ratios))
    default_boxes = []

    for m, fm_size in enumerate(feature_map_sizes):
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            default_boxes.append([
                cx,
                cy,
                scales[m],
                scales[m]
            ])

            default_boxes.append([
                cx,
                cy,
                math.sqrt(scales[m] * scales[m + 1]),
                math.sqrt(scales[m] * scales[m + 1])
            ])

            for ratio in aspect_ratios[m]:
                r = math.sqrt(ratio)
                default_boxes.append([
                    cx,
                    cy,
                    scales[m] * r,
                    scales[m] / r
                ])

                default_boxes.append([
                    cx,
                    cy,
                    scales[m] / r,
                    scales[m] * r
                ])

    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes
