import os
import re
import argparse
from datetime import datetime
from .yacs import CfgNode as CN


# -----------------------------------------------------------------------------
# global setting
# -----------------------------------------------------------------------------
cfg = CN()
cfg.run = 'run_1'
cfg.workers = 4
cfg.using_amp = True
cfg.imagenet_train_size = 313407
cfg.imagenet_test_size = 99964
cfg.errperc = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
cfg.record_dir = './doc/record'
cfg.result_dir = './doc/result'
cfg.tensor_Board = './tensorboard'
cfg.time_now = datetime.now().isoformat()
cfg.datacat = 'vmd'
# 'norm_select_218-Site_9A-Solibro', "norm_select_97-Site_22-SolFocus", "norm_select_93-Site_8-Kaneka"
# 'norm_select_87-Site_1B-Trina', 'norm_select_79-Site_7-First-Solar', 'norm_select_70-Site_3-BP-Solar'
# 'norm_select_69-Site_17-Sanyo'
cfg.dataname = "norm_select_218-Site_9A-Solibro"
cfg.dataroot = '/home/liuyunfei/data/powers/imgs'
cfg.networks = ['lan', 'resnet18', 'convnext', 'efficientnet', 'mobilenext', 'repvgg']

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()
cfg.train.lr = 0.0002
cfg.train.epoch = 40
cfg.train.imgsize = 72
cfg.train.optim = 'SGD'
cfg.train.momentum = 0.9
cfg.train.pretrain = False
cfg.train.log_interval = 100
cfg.train.weight_decay = 1e-4
cfg.train.batch_size = 160 # 160
cfg.train.clip_gradient = False
cfg.train.max_norm = 20.0


# -----------------------------------------------------------------------------
# test
# -----------------------------------------------------------------------------
cfg.test = CN()
cfg.test.epoch = -1
cfg.test.log_interval = 500
cfg.test.batch_size = 288

# 'resnet18', 'convnext', 'efficientnet', 'mobilenext', 'repvgg'
def make_cfg_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--iscap", type=bool, default=False, help='capsule network')
    parser.add_argument("--seed", type=int, default=0, help='seed for imagenet')
    parser.add_argument("--warm", type=int, default=5, help='warmup for imagenet')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='repvgg',
                    help='model architecture')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size per process (default: 128)')
    parser.add_argument('--device', default='cuda', type=str,
                    help='model GPU train')
    parser.add_argument('--capsules_num', default=6, type=int,
                    help='capsules number')
    parser.add_argument('--fcnum', default=6*3, type=int,
                    help='FC numclasses')
    parser.add_argument('--routing_iterations', default=2, type=int,
                    help='routing iterations')
    parser.add_argument('--backbone_channels', default=128, type=int,
                    help='backbone feature output channels')
    parser.add_argument('--backbone_hw', default=18, type=int,
                    help='backbone feature output width and heith')
    parser.add_argument('--capsules_tcc', default=16, type=int,
                    help='total capsules channels')
    parser.add_argument('--capsules_ecc', default=8, type=int,
                    help='each capsules channels')
    parser.add_argument('--capsules_k', default=9, type=int,
                    help='capsules kernel')
    parser.add_argument('--capsules_s', default=2, type=int,
                    help='capsules stride')
    parser.add_argument('--digit_dim', default=3, type=int,
                    help='digital dimention')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    cfg.heads = 'cap' if args.iscap else 'fc'

    cfg.arch = args.arch
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.datacat, cfg.dataname, cfg.arch, cfg.heads)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.datacat, cfg.dataname, cfg.arch, cfg.heads)
    cfg.data_path = os.path.join(cfg.dataroot, cfg.datacat, cfg.dataname)

    return cfg.clone(), args