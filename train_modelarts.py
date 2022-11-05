import os
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
import moxing as mox
mindspore.set_seed(1)


### Defines whether the task is a training environment or a debugging environment ###
def WorkEnvironment(environment): 
    if environment == 'train':
        workroot = '/home/work/user-job-dir'
    elif environment == 'debug':
        workroot = '/home/work' 
    print('current work mode:' + environment + ', workroot:' + workroot)
    return workroot

### Copy single dataset from obs to training image###
def ObsToEnv(obs_data_url, data_dir):
    try:     
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    return 

### Copy the output model to obs###
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
    return   

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
    parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default= WorkEnvironment('train') + '/data/')
    parser.add_argument('--train_url',
                        help='model folder to save/load',
                        default= WorkEnvironment('train') + '/model/')
    parser.add_argument(
                        '--device_target',type=str,default="Ascend",choices=['Ascend', 'CPU'],
                        help='device where the code will be implemented (default: Ascend),if to use the CPU on the Qizhi platform:device_target=CPU')

    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--pretrained", type=str, default="")
    # parser.add_argument("--pretrained", type=str, default="./segmenter_ms.ckpt")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_type", type=str, default='cos')
    parser.add_argument("--dataset", type=str, default="cityscapes",
                        help="Dataset.")
    parser.add_argument("--backbone", type=str, default='deit_base_distilled_patch16_384')
    parser.add_argument("--decoder", type=str, default="mask_transformer")
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--epochs", type=int, default=240)
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
    
    args = parse_args()
    
    # 运行模型，硬件设置
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=args.device_target)
    # context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target="GPU", device_id=device)
    ### defining the training environment
    environment = 'train'
    workroot = WorkEnvironment(environment)

    ###Initialize the data and model directories in the training image###
    data_dir = workroot + '/data'  
    train_dir = workroot + '/model'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    ### Copy the dataset from obs to the training image ###   
    ObsToEnv(args.data_url,data_dir)


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

    # dataset config
    batch_size = dataset_cfg["batch_size"]
    lr = dataset_cfg["learning_rate"]

    is_distributed=False
    # Create dataset
    dataset_dir = data_dir + "/cityscapes/"
    train_set, image_size, num_classes, class_weights = create_seg_dataset(
        args.dataset, dataset_dir, batch_size, is_distributed, is_train=True)

    steps_per_epoch = train_set.get_dataset_size()

    # model
    model_cfg['n_cls'] = 19
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
    # lr = get_cos_lr(args.lr, lr_min, args.epochs, steps_per_epoch)
    # lr = lr[begin_step:]
    power = 0.9
    total_step = args.epochs * steps_per_epoch
    lr = nn.polynomial_decay_lr(args.lr, lr_min, total_step, steps_per_epoch, args.epochs, power)
    # # Optimizer
    # 
    # tt=network.trainable_params()
    opt = nn.SGD(network.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0, nesterov=True)
    # opt = nn.Adam(network.trainable_params(), learning_rate=lr, weight_decay=0.0001, use_nesterov=True)
    # param_dict = network.parameters_dict()
    # loss_scale = 1024
    # loss_scale_manager = FixedLossScaleManager(loss_scale, False)
    # profiler = Profiler(output_path = './profiler_data')  # add

    network.set_train(True)
    # Create model
    model = Model(network, loss_fn=loss, optimizer=opt, amp_level="O2", keep_batchnorm_fp32=True)
    # model = Model(network, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager, amp_level="O3",
    #               keep_batchnorm_fp32=True)

    # Callbacks
    time_loss_cb = TimeLossMonitor(lr_init=lr)
    cb = [time_loss_cb]
    # Save-checkpoint callback
    save_checkpoint_epochs = 1 #保存频率
    keep_checkpoint_max = 20 #最多保存的数量
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * save_checkpoint_epochs,
                                   keep_checkpoint_max=keep_checkpoint_max)

    ckpt_cb = ModelCheckpoint(prefix=f"segmenter",
                              directory=train_dir,
                              config=ckpt_config)
    cb.append(ckpt_cb)

    # Self-defined callbacks
    # eval = False
    # if eval:
    #     val_set, _, _, _ = create_seg_dataset(
    #     args.dataset, args.data_path, batchsize=1, run_distribute=False, is_train=False)

    #     num_classes = 19
    #     eval_start = 0
    #     interval = 1
    #     eval_cb = SegEvalCallback(val_set, network, num_classes, start_epoch=eval_start,
    #                               save_path=local_train_url, interval=interval)
    #     cb.append(eval_cb)

    model.train(args.epochs, train_set, callbacks=cb, dataset_sink_mode=True)
    # profiler.analyse()  # add

    ###and download it in the training task corresponding to the Qizhi platform
    EnvToObs(train_dir, args.train_url)

        

if __name__ == "__main__":
    main()