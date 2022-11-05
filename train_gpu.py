import os
import ast
import time
import argparse
import numpy as np
import mindspore
from mindspore import context
from mindspore.train.model import Model
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from mindspore.profiler import Profiler
from config.config import *
from src.utils import *
from src.loss import SoftmaxCrossEntropyLoss
from src.dataset.dataset_generator import create_seg_dataset
from src.factory import create_segmenter
from src.dataset.utils import *
from src.optimizer import *
from src.lr_generator import *
from src.callback import TimeLossMonitor, SegEvalCallback
mindspore.set_seed(1)

class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss

def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description="Mindspore Segmenter Training Configurations.")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--pretrained", type=str, default="")
    # parser.add_argument("--pretrained", type=str, default="./segmenter_ms.ckpt")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_type", type=str, default='cos')
    parser.add_argument("--data_path", type=str, default="/work/ai_lab/miner/data/cityscapes/")
    parser.add_argument("--dataset", type=str, default="cityscapes",
                        help="Dataset.")
    parser.add_argument("--backbone", type=str, default='vit_small_patch16_384')
    parser.add_argument("--decoder", type=str, default="mask_transformer")
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False,
                        help="Run on ModelArts or offline machines.")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")
    parser.add_argument("--eval", type=ast.literal_eval, default=False)
    return parser.parse_args()


def main():
    """Training process."""
    set_seed(1)
    args = parse_args()

    if args.run_distribute:
        init()
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = int(os.getenv("RANK_SIZE"))
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_num=device_num)
    else:
        device_id = 0

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id = 2)


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

    model_cfg["image_size"] = [768,768]
    # model_cfg["image_size"] = 768
    model_cfg["backbone"] = args.backbone
    model_cfg["dropout"] = args.dropout
    model_cfg["drop_path_rate"] = args.drop_path
    decoder_cfg["name"] = "mask_transformer"
    model_cfg["decoder"] = decoder_cfg
    model_cfg['n_cls'] = 19

    # dataset config
    batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]

    # Create dataset
    train_set, image_size, num_classes, class_weights = create_seg_dataset(
        args.dataset, args.data_path, batch_size, args.run_distribute, is_train=True)

    steps_per_epoch = train_set.get_dataset_size()

    # model
    network = create_segmenter(model_cfg)

    # load pretrained model
    if args.pretrained:
        param_dict = load_checkpoint(args.pretrained)
        load_param_into_net(network, param_dict)
        print('load_model {} success'.format(args.pretrained))
    
    # loss
    num_classes=19
    ignore_label=255
    loss = SoftmaxCrossEntropyLoss(num_classes, ignore_label)
    # Learning rate adjustment.
    begin_epoch = 0
    begin_step = begin_epoch * steps_per_epoch
    lr_min = 1.0e-05
    print("learning rate")
    print(args.lr)

    power = 0.9
    total_step = args.epochs * steps_per_epoch
    lr = nn.polynomial_decay_lr(args.lr, lr_min, total_step, steps_per_epoch, args.epochs, power)
    # # Optimizer
    opt = nn.SGD(network.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0, nesterov=True)
    # opt = nn.Adam(network.trainable_params(), learning_rate=args.lr, weight_decay=0.0001, use_nesterov=True)

    network.set_train(True)
    # Create model
    model = Model(network, loss_fn=loss, optimizer=opt, amp_level="O2", keep_batchnorm_fp32=True)
 
    # Callbacks
    time_loss_cb = TimeLossMonitor(lr_init=lr)
    cb = [time_loss_cb]
    # Save-checkpoint callback
    save_checkpoint_epochs = 1 #保存频率
    keep_checkpoint_max = 10 #最多保存的数量
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * save_checkpoint_epochs,
                                   keep_checkpoint_max=keep_checkpoint_max)
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    local_train_url = "./ckpt/"+time_now
    ckpt_cb = ModelCheckpoint(prefix=f"segmenter",
                              directory=local_train_url,
                              config=ckpt_config)
    cb.append(ckpt_cb)

    # Self-defined callbacks
    eval = False
    if args.eval:
        val_set, _, _, _ = create_seg_dataset(
        args.dataset, args.data_path, batchsize=1, run_distribute=False, is_train=False)

        num_classes = 19
        eval_start = 0
        interval = 1
        eval_cb = SegEvalCallback(val_set, network, num_classes, start_epoch=eval_start,
                                  save_path=local_train_url, interval=interval)
        cb.append(eval_cb)

    model.train(args.epochs, train_set, callbacks=cb, dataset_sink_mode=True)

    # if args.modelarts:
    #     import moxing as mox
    #     mox.file.copy_parallel(local_train_url, args.train_url)


        

            

if __name__ == "__main__":
    main()