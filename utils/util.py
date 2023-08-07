import os
import shutil
import torch
import numpy as np
import random
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.distributed as dist


def save_checkpoint(state, is_best, cfg, args, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    model_dir = os.path.join(cfg.result_dir, cfg.run + '_' + str(args.seed))
    os.makedirs(model_dir, exist_ok=True)
    filename = os.path.join(model_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth.tar'))

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max , eta_min = 0, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters <= self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            if self.iters == self.warmup_iters:
                self.iters = 0
                self.warmup = None
            return
        
        # cos policy

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=1000, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_v1(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # import pdb; pdb.set_trace()
        loss = (- targets * log_probs).mean(0).sum()
        return loss
    
    def forward_v2(self, inputs, targets):
        probs = self.logsoftmax(inputs)
        targets = torch.zeros(probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = nn.KLDivLoss()(probs, targets)
        return loss

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        return self.forward_v1(inputs, targets)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def reduce_value(value):
    world_size = get_world_size()
    rt = value.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

class CompleLoss(nn.Module):
    def __init__(self, percs=[0.3, 0.25, 0.2, 0.15, 0.1, 0.05], iscap=None):
        super(CompleLoss, self).__init__()
        self.d1ap = percs[0]
        self.h6ap = percs[1]
        self.h1ap = percs[2]
        self.min30ap = percs[3]
        self.min15ap = percs[4]
        self.min5ap = percs[5]
        self.register_buffer('weights', torch.Tensor([
            [1+self.d1ap, 1, 1-self.d1ap],
            [1+self.h6ap, 1, 1-self.h6ap],
            [1+self.h1ap, 1, 1-self.h1ap],
            [1+self.min30ap, 1, 1-self.min30ap],
            [1+self.min15ap, 1, 1-self.min15ap],
            [1+self.min5ap, 1, 1-self.min5ap]]))
        self.capmseloss = nn.MSELoss(reduce=False)
        self.fcmseloss = nn.MSELoss(reduction='mean')
        self.iscap = iscap
    def _cap_forward(self, pres, tars):  # tars=bx6 -> bx6x3, pres=bx6x3
        b, tem = tars.shape
        tars = tars.unsqueeze(-1)
        tars = tars * self.weights
        loss = torch.mean(torch.sum(self.capmseloss(pres, tars), dim=-1))
        return loss

    def _fc_forward(self, pres, tars):
        b, _ = tars.shape
        tars = (tars.unsqueeze(-1) * self.weights).view(b, -1)
        loss = self.fcmseloss(pres, tars)
        return loss

    def forward(self, pres, tars):
        if self.iscap:
            return self._cap_forward(pres, tars)
        else:
            return self._fc_forward(pres, tars)