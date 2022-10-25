import tensorflow as tf
from tensorflow.keras import Model
from .common import get_backbone, create_extra_layers, create_heads


def ssd(backbone, input_shape, num_classes, aspect_ratios):
    input_size = input_shape[0]
    base_model, feature_names = get_backbone(backbone, input_size)
    extra_layers = create_extra_layers(backbone)

    features = []
    for name in feature_names:
        features.append(base_model.get_layer(name).output)

    x = base_model.output
    for layer in extra_layers:
        x = layer(x)
        features.append(x)

    confs = []
    locs = []
    conf_head_layers, loc_head_layers = create_heads(
        backbone, num_classes, aspect_ratios)
    for i, feature in enumerate(features):
        conf = conf_head_layers[i](feature)
        print(type(loc_head_layers[i]))
        loc = loc_head_layers[i](feature)

        print(conf.shape)
        confs.append(tf.reshape(conf, [tf.shape(conf)[0], -1, num_classes]))
        locs.append(tf.reshape(loc, [tf.shape(loc)[0], -1, 4]))
    return Model(base_model.input, [confs, locs])
