import time
import os
import math
import sys
from typing import List, Tuple, Union, Callable, Dict

import numpy as np
import tensorflow as tf

from meghnad.core.cv.obj_det.src.tf.data_loader import TFObjDetDataLoader
from meghnad.core.cv.obj_det.src.tf.model_loader.losses import Loss
from meghnad.core.cv.obj_det.cfg import ObjDetConfig

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header

from meghnad.core.cv.obj_det.src.tf.trn.select_model import TFObjDetSelectModel
from meghnad.core.cv.obj_det.src.tf.trn.eval import TFObjDetEval
from meghnad.core.cv.obj_det.src.tf.trn.trn_utils import get_optimizer


__all__ = ['TFObjDetTrn']


log = Log()


@method_header(
    description="""Process a training step.

    Parameters
    ----------
    imgs : tf.Tensor
        Images for training. A tensor has shape of [N, H, W, C]
    gt_confs : tf.Tensor
        Classification targets. A tensor has shape of [B, num_default]
    gt_locs : tf.Tensor
        Regression targets. A tensor has shape of [B, num_default, 4]
    model : tf.keras.Model
        An instance of tf.keras.Model
    criterion : function
        Loss function
    optimizer : Optimizer class
        Optimizer for updating weights
    weight_decay : float
        Weights decay

    Returns
    -------
    [loss, conf_loss, loc_loss, l2_loss]
        Returns a list of losses.
    """)
@tf.function
def _train_step(
        imgs: tf.Tensor,
        gt_confs: tf.Tensor,
        gt_locs: tf.Tensor,
        model: tf.keras.Model,
        criterion: Union[tf.keras.losses.Loss, Callable],
        optimizer: str,
        weight_decay: float = 1e-5) -> Tuple[float]:

    with tf.GradientTape() as tape:
        confs, locs = model(imgs)

        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs)

        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
        l2_loss = weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    # Normalize gradients
    gradients = [tf.clip_by_norm(grad, 0.2) for grad in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


@method_header(
    description="""Returns configs from given settings

    Parameters
    ----------
    settings : List[str]
        A list of string represents settings.

    Returns
    -------
    [model_cfgs]
        A list of string represents corresponding model configs
    """)
def load_config_from_settings(settings: List[str]) -> Tuple[List, List]:

    settings = [f'{setting}_models' for setting in settings]
    cfg_obj = ObjDetConfig()
    data_cfg = cfg_obj.get_data_cfg()

    model_cfgs = []
    data_cfgs = []
    for setting in settings:
        model_names = cfg_obj.get_model_settings(setting)
        for model_name in model_names:
            model_cfg = cfg_obj.get_model_cfg(model_name)
            model_cfgs.append(model_cfg)
            data_cfgs.append(data_cfg)
    return model_cfgs, data_cfgs


@class_header(
    description='''
        Class for object detection model training''')
class TFObjDetTrn:
    def __init__(self, settings: List[str]) -> None:
        self.settings = settings
        self.model_cfgs, self.data_cfgs = load_config_from_settings(settings)
        self.model_selection = TFObjDetSelectModel(self.model_cfgs)
        self.data_loaders = []
        self.best_model_path = None

    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data (should point to the file in case of a single file, should point to
                the directory in case data exists in multiple files in a directory structure)
                ''')
    def config_connectors(self, data_path: str, augmentations: Dict = None) -> None:
        self.data_loaders = [TFObjDetDataLoader(data_path, data_cfg, model_cfg, augmentations)
                             for data_cfg, model_cfg in zip(self.data_cfgs, self.model_cfgs)]

    @method_header(
        description='''
                Function to set training configurations and start training.''',
        arguments='''
                epochs: set epochs for the training by default it is 10
                checkpoint_dir: directory from where the checkpoints should be loaded
                logdir: directory where the logs should be saved
                resume_path: The path/checkpoint from where the training should be resumed
                print_every: an argument to specify when the function should print or after how many epochs
                ''')
    def trn(self,
              epochs: int = 10,
              checkpoint_dir: str = './checkpoints',
              logdir: str = './training_logs',
              resume_path: str = None,
              print_every: int = 10,
              **hyp) -> Tuple:
        try:
            epochs = int(epochs)
            if epochs <= 0:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__, "Epochs value must be a positive integer")
                return ret_values.IXO_RET_INVALID_INPUTS

        except ValueError:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Epochs value must be a positive integer")
            return ret_values.IXO_RET_INVALID_INPUTS

        best_map_over_all_models = -1.0
        for i, model in enumerate(self.model_selection.models):
            data_loader = self.data_loaders[i]
            model_cfg = self.model_cfgs[i]
            opt = hyp.get('optimizer', 'Adam')
            weight_decay = hyp.get('weight_decay', 1e-5)

            optimizer = get_optimizer(opt)
            criterion = Loss(
                model_cfg['neg_ratio'], model_cfg['num_classes'])
            evaluator = TFObjDetEval(model)

            model_name = model_cfg['arch']
            log_dir = os.path.join(logdir, model_name)

            # Found a checkpoint
            ckpt = tf.train.Checkpoint(
                model=model, optimizer=optimizer, start_epoch=tf.Variable(0))
            if resume_path:
                ckpt.read(resume_path)
                print(f'Resume training from {resume_path}')
            else:
                if checkpoint_dir and os.path.isdir(checkpoint_dir):
                    ckpt_filenames = os.listdir(checkpoint_dir)
                    prefix = f'{model_name}_last.ckpt'
                    for filename in ckpt_filenames:
                        if filename.startswith(prefix):
                            ckpt_path = os.path.join(
                                checkpoint_dir, prefix)
                            ckpt.read(ckpt_path)
                            print(f'Resume training from {ckpt_path}')
                            break

            # Setup summary writers
            train_log_dir = os.path.join(log_dir, 'train')
            val_log_dir = os.path.join(log_dir, 'val')
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)

            # Setup learning rate scheduler
            base_lr = float(optimizer.learning_rate.numpy())
            warmup_learning_rate = base_lr / 6
            warmup_steps = 2000
            optimizer.learning_rate.assign(warmup_learning_rate)
            steps_per_epoch = max(
                1, data_loader.train_size // data_loader.batch_size)
            total_steps = epochs * steps_per_epoch
            global_step = 0

            # TODO: replace print by Log?
            print('Steps per epoch', steps_per_epoch)
            print('Total steps', total_steps)

            # Start training
            start_epoch = ckpt.start_epoch.numpy()
            for epoch in range(start_epoch, epochs):
                avg_loss = 0.0
                avg_conf_loss = 0.0
                avg_loc_loss = 0.0
                avg_l2_loss = 0.0
                start = time.time()
                for i, (imgs, gt_confs, gt_locs) in enumerate(data_loader.train_dataset):
                    # Forward + Backward
                    loss, conf_loss, loc_loss, l2_loss = _train_step(
                        imgs, gt_confs, gt_locs,
                        model, criterion, optimizer, weight_decay
                    )

                    # Compute average losses
                    avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
                    avg_conf_loss = (avg_conf_loss * i +
                                     conf_loss.numpy()) / (i + 1)
                    avg_loc_loss = (avg_loc_loss * i +
                                    loc_loss.numpy()) / (i + 1)
                    avg_l2_loss = (avg_l2_loss * i + l2_loss.numpy()) / (i + 1)
                    if (i + 1) % print_every == 0:
                        print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f} L2 Loss {:.4f}'.format(
                            epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss, avg_l2_loss))

                    # Learning rate scheduler
                    global_step = epoch * steps_per_epoch + i + 1
                    if global_step <= warmup_steps:
                        slope = (base_lr - warmup_learning_rate) / warmup_steps
                        new_lr = warmup_learning_rate + slope * \
                            tf.cast(global_step, tf.float32)
                        optimizer.learning_rate.assign(new_lr)
                    else:
                        new_lr = 0.5 * base_lr * (1 + tf.cos(
                            math.pi *
                            (tf.cast(i + 1, tf.float32) - warmup_steps
                             ) / float(total_steps - warmup_steps)))
                        optimizer.learning_rate.assign(new_lr)
                print('Current learning rate:',
                      optimizer.learning_rate.numpy())

                # Start evaluation at the end of epoch
                print('Evaluating...')
                map, map50 = evaluator.eval(data_loader)

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', avg_loss, step=epoch)
                    tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
                    tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

                with val_summary_writer.as_default():
                    tf.summary.scalar('mAP', map, step=epoch)

                # Checkpoint
                ckpt.start_epoch.assign_add(1)
                save_path = ckpt.write(os.path.join(
                    checkpoint_dir, f'{model_name}_last.ckpt'))
                print("Saved checkpoint for epoch {}: {}".format(
                    int(ckpt.start_epoch), save_path))

                # Save the best
                if map > best_map_over_all_models:
                    best_map_over_all_models = map

                    self.best_model_path = os.path.join(
                        checkpoint_dir, f'best_saved_model', model_name)

                    tf.saved_model.save(model, self.best_model_path)
                    print(f'Saved the best model as {self.best_model_path}')

                    # save the corresponding default bounding boxes for inference
                    metadata_path = os.path.join(checkpoint_dir,
                                                 f'best_saved_model', model_name, 'metadata.npz')
                    np.savez(metadata_path, default_boxes=data_loader.default_boxes,
                             input_shape=data_loader.input_shape)
                    print(f'Saved model metadata as {metadata_path}')

            return ret_values.IXO_RET_SUCCESS, self.best_model_path