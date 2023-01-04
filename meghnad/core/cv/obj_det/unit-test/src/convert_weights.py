import torch
from meghnad.core.cv.obj_det.src.pytorch.train.utils.common import get_meghnad_repo_dir
from meghnad.repo.obj_det.yolov7.models.experimental import attempt_load
from meghnad.repo.obj_det.yolov7.models.yolo import Model
from meghnad.repo.obj_det.yolov7.utils.autoanchor import check_anchors
from meghnad.repo.obj_det.yolov7.utils.datasets import create_dataloader
from meghnad.repo.obj_det.yolov7.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from meghnad.repo.obj_det.yolov7.utils.google_utils import attempt_download
from meghnad.repo.obj_det.yolov7.utils.loss import ComputeLoss, ComputeLossOTA
from meghnad.repo.obj_det.yolov7.utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from meghnad.repo.obj_det.yolov7.utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from meghnad.repo.obj_det.yolov7.utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume


def main():
    weights = 'yolov7.pt'
    state_dict = torch.load(weights, map_location='cpu')
    print(state_dict.keys())


if __name__ == '__main__':
    main()
