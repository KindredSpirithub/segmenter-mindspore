import os
import time
import ast
import argparse
import numpy as np
import mindspore
from mindspore import context
from mindspore.nn import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from config.config import *
from src.utils import *
# from loss.loss import SoftmaxCrossEntropyLoss
from src.dataset.dataset_generator import create_seg_dataset
from src.factory import create_segmenter
from src.segmenter import SegmenterWithLossCell
from src.utils import get_confusion_matrix, resize
mindspore.set_seed(1)

def get_cos_lr(lr_max, lr_min, total_epoch, spe):
    """Get learning rates decaying in cosine annealing mode."""
    lr_min = lr_min
    lr_max = lr_max
    lrs = []
    total_step = spe * total_epoch
    for i in range(total_step):
        lrs.append(lr_min + (lr_max - lr_min) * (1 + np.cos(i * np.pi / total_step)) / 2)
    return lrs

def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description="Mindspore Segmenter Training Configurations.")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt/20220727-0815/segmenter_102_371.ckpt")
    parser.add_argument("--data_path", type=str, default="/work/ai_lab/miner/data/cityscapes/")
    parser.add_argument("--dataset", type=str, default="cityscapes",
                        help="Dataset.")
    parser.add_argument("--backbone", type=str, default='vit_tiny_patch16_384')
    parser.add_argument("--decoder", type=str, default="mask_transformer")
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--resume", type=bool, default=False)

    return parser.parse_args()

class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss

def main():
    """Training process."""
    set_seed(1)
    # 运行模型，硬件设置
    device = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target="GPU", device_id=device)
    args = parse_args()

    # set up configuration
    cfg = load_config()

    model_cfg = cfg["model"][args.backbone]
    dataset_cfg = cfg["dataset"][args.dataset]
    decoder_cfg = cfg["decoder"]["mask_transformer"]

    # model config
    im_size = dataset_cfg["im_size"]
    crop_size = dataset_cfg.get("crop_size", im_size)
    window_size = dataset_cfg.get("window_size", im_size)
    window_stride = dataset_cfg.get("window_stride", im_size)

    # model_cfg["image_size"] = 768
    model_cfg["image_size"] = [768,768]
    model_cfg["backbone"] = args.backbone
    model_cfg["dropout"] = args.dropout
    model_cfg["drop_path_rate"] = args.drop_path
    decoder_cfg["name"] = "mask_transformer"
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    batch_size = 1  # 必须是1
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]
    # 如果有输入参数，就用输入的参数值

    eval_freq = dataset_cfg.get("eval_freq", 1)

    rank=0
    group_size=1
    is_distributed=False
    # init multicards training
    if is_distributed:
        init()
        rank = get_rank()
        group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=group_size)

    # val set
    val_set, image_size, num_classes, class_weights = create_seg_dataset(
        args.dataset, args.data_path, batch_size, is_distributed, is_train=False)
    
    steps_num = val_set.get_dataset_size()
    val_loader = val_set.create_dict_iterator()
    # model
    model_cfg['n_cls'] = 19
    network = create_segmenter(model_cfg)


    param_dict = load_checkpoint(ckpt_file_name=args.ckpt_dir)
    load_param_into_net(network, param_dict)
    # loss
    num_classes=19
    ignore_label=255
    # criterion = SoftmaxCrossEntropyLoss(num_classes, ignore_label)
    # validation
    print("start eval......")
    network.set_train(False)
    confusion_matrix = np.zeros((num_classes, num_classes))
    count = 0
    ori_shape=[1024,2048]
    zeros = ops.Zeros()
    seg_map = zeros((num_classes, ori_shape[0], ori_shape[1]), mindspore.float32)
    for step_idx, data in enumerate(val_loader):
        images = data['image']
        seg_gt = data['label']
        shape = seg_gt.shape
        resize_bilinear = nn.ResizeBilinear()
        images = resize_bilinear(images, size=(768, 1536))
        # images = resize(images, window_size) # 等比例缩放,[768,1532]
        flip=False
        windows = sliding_window(images, flip, window_size, window_stride) 
        tt = windows['crop']
        stack = ops.Stack()
        yy = stack(tt)
        crops = stack(windows.pop("crop")).reshape((batch_size*3, 3, window_size, window_size))
        # crops = stack(windows.pop("crop"))[:, 0] # [3,3,768,768]
        # crops = mindspore.ops.Stack(windows.pop("crop"))[:, 0] # [3,3,768,768]
        B = len(crops) # 3*batch_size
        WB = batch_size
        seg_maps = zeros((B, num_classes, window_size, window_size), mindspore.float32)
        for i in range(0, B, WB):
            seg_maps[i : i + WB] = network(crops[i : i + WB])
        windows["seg_maps"] = seg_maps
        pred = merge_windows(windows, window_size, ori_shape)

        confusion_matrix += get_confusion_matrix(seg_gt, pred, shape, num_classes, 255)
        count += 1
        if step_idx  % 20 == 0:
            print("eval progress : {}".format(float(step_idx)/float(steps_num)))
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    iou_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_iou = iou_array.mean()
    print("miou:")
    print(mean_iou)
           

if __name__ == "__main__":
    main()