import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
from pathlib import Path
from collections import OrderedDict
sys.path.append("..")
from utils import *
from datasets import dataloader
from networks import create_net
from configs import make_cfg_args


def cap_interval(x, inv):
	x = np.array(x)
	ratio_up, ration_do = 1+inv, 1-inv
	upper, down = x*ratio_up, x*ration_do
	return upper.tolist(), down.tolist()

def fit_interval(x, y, ci=90):
    x, y = np.array(x), np.array(y)
    alpha = 1 - ci / 100
    n = len(x)
    Sxx = np.sum(x ** 2) - np.sum(x) ** 2 / n
    Sxy = np.sum(x * y) - np.sum(x) * np.sum(y) / n
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    b = Sxy / Sxx
    a = mean_y - b * mean_x
    def fit(xx):
        return a + b * xx
    residuals = y - fit(x)
    var_res = np.sum(residuals ** 2) / (n - 2)
    sd_res = np.sqrt(var_res)
    se_b = sd_res / np.sqrt(Sxx)
    se_a = sd_res * np.sqrt(np.sum(x ** 2) / (n * Sxx))
    df = n - 2  # degrees of freedom
    tval = stats.t.isf(alpha / 2., df)  # appropriate t value
    def se_fit(x):
        return sd_res * np.sqrt(1. / n + (x - mean_x) ** 2 / Sxx)
    upper = fit(x) + tval * se_fit(x)
    down = fit(x) - tval * se_fit(x)
    return upper.tolist(), down.tolist()

class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def metrics(forecasts, actuals):
    with torch.no_grad():
        n = actuals.size(0)
        assert n >= 1, 'Batch size is greater than or equal to 1'
        eps = 1e-8
        forecasts = forecasts.squeeze().float()
        actuals = actuals.float()
        mean_A = torch.mean(actuals)
        mean_F = torch.mean(forecasts)
        fF = forecasts - mean_F
        aA = actuals - mean_A
        fa = forecasts - actuals
        fa_ = torch.abs(forecasts) + torch.abs(actuals)
        R = torch.einsum('i,i->', [fF, aA]) / torch.sqrt(torch.einsum('i->', [fF**2]) * torch.einsum('i->', [aA**2]) + eps)
        RMSE = torch.sqrt(torch.einsum('i->', [fa**2]) / n)
        MAE = torch.einsum('i->', [torch.abs(fa)]) / n
        SMAPE = torch.einsum('i->', [torch.abs(fa) / (eps + fa_ / 2)]) / n

        return R, RMSE, MAE, SMAPE

def test(cfg, args, model, val_loader, path_model):
    path_model = Path(path_model)
    current_file = os.path.abspath('.')
    norpath = Path(current_file) / cfg.datacat / cfg.dataname / cfg.arch / cfg.heads
    norpath.mkdir(parents=True, exist_ok=True)
    recordict = OrderedDict(min5t=[], min5p=[], min5tup=[], min5tdo=[], min5ppup=[], min5ppdo=[], min5pup=[], min5pdo=[], min5r=[], min5rmse=[], min5mae=[], min5smape=[],
                            min15t=[], min15p=[], min15tup=[], min15tdo=[], min15ppup=[], min15ppdo=[], min15pup=[], min15pdo=[], min15r=[], min15rmse=[], min15mae=[], min15smape=[],
                            min30t=[], min30p=[], min30tup=[], min30tdo=[], min30ppup=[], min30ppdo=[], min30pup=[], min30pdo=[], min30r=[], min30rmse=[], min30mae=[], min30smape=[],
                            h1t=[], h1p=[], h1tup=[], h1tdo=[], h1ppup=[], h1ppdo=[], h1pup=[], h1pdo=[], h1r=[], h1rmse=[], h1mae=[], h1smape=[],
                            h6t=[], h6p=[], h6tup=[], h6tdo=[], h6ppup=[], h6ppdo=[], h6pup=[], h6pdo=[], h6r=[], h6rmse=[], h6mae=[], h6smape=[],
                            d1t=[], d1p=[], d1tup=[], d1tdo=[], d1ppup=[], d1ppdo=[], d1pup=[], d1pdo=[], d1r=[], d1rmse=[], d1mae=[], d1smape=[]
                            )
    # activate power 1 day
    ap1d_R_meter = AverageMeter('ap1d_R', ':6.2f')
    ap1d_RMSE_meter = AverageMeter('ap1d_RMSE', ':6.2f')
    ap1d_MAE_meter = AverageMeter('ap1d_MAE', ':6.2f')
    ap1d_SMAPE_meter = AverageMeter('ap1d_SMAPE', ':6.2f')
    # activate power 6 hour
    ap6h_R_meter = AverageMeter('ap6h_R', ':6.2f')
    ap6h_RMSE_meter = AverageMeter('ap6h_RMSE', ':6.2f')
    ap6h_MAE_meter = AverageMeter('ap6h_MAE', ':6.2f')
    ap6h_SMAPE_meter = AverageMeter('ap6h_SMAPE', ':6.2f')
    # activate power 1 hour
    ap1h_R_meter = AverageMeter('ap1h_R', ':6.2f')
    ap1h_RMSE_meter = AverageMeter('ap1h_RMSE', ':6.2f')
    ap1h_MAE_meter = AverageMeter('ap1h_MAE', ':6.2f')
    ap1h_SMAPE_meter = AverageMeter('ap1h_SMAPE', ':6.2f')
    # activate power 30min
    ap30min_R_meter = AverageMeter('ap30min_R', ':6.2f')
    ap30min_RMSE_meter = AverageMeter('ap30min_RMSE', ':6.2f')
    ap30min_MAE_meter = AverageMeter('ap30min_MAE', ':6.2f')
    ap30min_SMAPE_meter = AverageMeter('ap30min_SMAPE', ':6.2f')
    # activate power 15min
    ap15min_R_meter = AverageMeter('ap15min_R', ':6.2f')
    ap15min_RMSE_meter = AverageMeter('ap15min_RMSE', ':6.2f')
    ap15min_MAE_meter = AverageMeter('ap15min_MAE', ':6.2f')
    ap15min_SMAPE_meter = AverageMeter('ap15min_SMAPE', ':6.2f')
    # activate power 5min
    ap5min_R_meter = AverageMeter('ap5min_R', ':6.2f')
    ap5min_RMSE_meter = AverageMeter('ap5min_RMSE', ':6.2f')
    ap5min_MAE_meter = AverageMeter('ap5min_MAE', ':6.2f')
    ap5min_SMAPE_meter = AverageMeter('ap5min_SMAPE', ':6.2f')
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(val_loader), desc='Processing', total=len(val_loader)):
            data, target = data.to(device=torch.device(args.device), non_blocking=True), torch.stack(target, dim=1).to(device=torch.device(args.device), non_blocking=True).float()
            output = model(data)
            if args.iscap:
                ap1d_R, ap1d_RMSE, ap1d_MAE, ap1d_SMAPE = metrics(output[:, 0, 1], target[:, 0])
                ap6h_R, ap6h_RMSE, ap6h_MAE, ap6h_SMAPE = metrics(output[:, 1, 1], target[:, 1])
                ap1h_R, ap1h_RMSE, ap1h_MAE, ap1h_SMAPE = metrics(output[:, 2, 1], target[:, 2])
                ap30min_R, ap30min_RMSE, ap30min_MAE, ap30min_SMAPE = metrics(output[:, 3, 1], target[:, 3])
                ap15min_R, ap15min_RMSE, ap15min_MAE, ap15min_SMAPE = metrics(output[:, 4, 1], target[:, 4])
                ap5min_R, ap5min_RMSE, ap5min_MAE, ap5min_SMAPE = metrics(output[:, 5, 1], target[:, 5])
            else:
                ap1d_R, ap1d_RMSE, ap1d_MAE, ap1d_SMAPE = metrics(output[:, 1], target[:, 0])
                ap6h_R, ap6h_RMSE, ap6h_MAE, ap6h_SMAPE = metrics(output[:, 4], target[:, 1])
                ap1h_R, ap1h_RMSE, ap1h_MAE, ap1h_SMAPE = metrics(output[:, 7], target[:, 2])
                ap30min_R, ap30min_RMSE, ap30min_MAE, ap30min_SMAPE = metrics(output[:, 10], target[:, 3])
                ap15min_R, ap15min_RMSE, ap15min_MAE, ap15min_SMAPE = metrics(output[:, 13], target[:, 4])
                ap5min_R, ap5min_RMSE, ap5min_MAE, ap5min_SMAPE = metrics(output[:, 16], target[:, 5])
            ap1d_R_meter.update(ap1d_R, data.size(0))
            ap1d_RMSE_meter.update(ap1d_RMSE, data.size(0))
            ap1d_MAE_meter.update(ap1d_MAE, data.size(0))
            ap1d_SMAPE_meter.update(ap1d_SMAPE, data.size(0))
            ap6h_R_meter.update(ap6h_R, data.size(0))
            ap6h_RMSE_meter.update(ap6h_RMSE, data.size(0))
            ap6h_MAE_meter.update(ap6h_MAE, data.size(0))
            ap6h_SMAPE_meter.update(ap6h_SMAPE, data.size(0))
            ap1h_R_meter.update(ap1h_R, data.size(0))
            ap1h_RMSE_meter.update(ap1h_RMSE, data.size(0))
            ap1h_MAE_meter.update(ap1h_MAE, data.size(0))
            ap1h_SMAPE_meter.update(ap1h_SMAPE, data.size(0))
            ap30min_R_meter.update(ap30min_R, data.size(0))
            ap30min_RMSE_meter.update(ap30min_RMSE, data.size(0))
            ap30min_MAE_meter.update(ap30min_MAE, data.size(0))
            ap30min_SMAPE_meter.update(ap30min_SMAPE, data.size(0))
            ap15min_R_meter.update(ap15min_R, data.size(0))
            ap15min_RMSE_meter.update(ap15min_RMSE, data.size(0))
            ap15min_MAE_meter.update(ap15min_MAE, data.size(0))
            ap15min_SMAPE_meter.update(ap15min_SMAPE, data.size(0))
            ap5min_R_meter.update(ap5min_R, data.size(0))
            ap5min_RMSE_meter.update(ap5min_RMSE, data.size(0))
            ap5min_MAE_meter.update(ap5min_MAE, data.size(0))
            ap5min_SMAPE_meter.update(ap5min_SMAPE, data.size(0))
            # for recording
            min5t = target[:, 0].cpu().tolist()
            min5p = output[:, 0, 1].cpu().tolist() if args.iscap else output[:, 1].cpu().tolist()
            recordict['min5t'] = recordict['min5t'] + min5t
            recordict['min5p'] = recordict['min5p'] + min5p
            # min5up, min5do = fit_interval(x=[i for i in range(len(min5t))], y=min5t)
            min5up, min5do = stats.t.interval(0.9, len(min5t)-1, loc=min5t, scale=stats.sem(min5t))
            min5up, min5do = min5up.tolist(), min5do.tolist()
            recordict['min5tup'] = recordict['min5tup'] + min5up
            recordict['min5tdo'] = recordict['min5tdo'] + min5do
            min5ppup, min5ppdo = cap_interval(x=min5t, inv=cfg.errperc[-1])
            recordict['min5ppup'] = recordict['min5ppup'] + min5ppup
            recordict['min5ppdo'] = recordict['min5ppdo'] + min5ppdo
            if args.iscap:
                recordict['min5pup'] = recordict['min5pup'] + output[:, 0, 0].cpu().tolist()
                recordict['min5pdo'] = recordict['min5pdo'] + output[:, 0, 2].cpu().tolist()
            else:
                recordict['min5pup'] = recordict['min5pup'] + output[:, 0].cpu().tolist()
                recordict['min5pdo'] = recordict['min5pdo'] + output[:, 2].cpu().tolist()
            min15t = target[:, 1].cpu().tolist()
            min15p = output[:, 1, 1].cpu().tolist() if args.iscap else output[:, 4].cpu().tolist()
            recordict['min15t'] = recordict['min15t'] + min15t
            recordict['min15p'] = recordict['min15p'] + min15p
            # min15up, min15do = fit_interval(x=[i for i in range(len(min15t))], y=min15t)
            min15up, min15do = stats.t.interval(0.9, len(min15t)-1, loc=min15t, scale=stats.sem(min15t))
            min15up, min15do = min15up.tolist(), min15do.tolist()
            recordict['min15tup'] = recordict['min15tup'] + min15up
            recordict['min15tdo'] = recordict['min15tdo'] + min15do
            min15ppup, min15ppdo = cap_interval(x=min15t, inv=cfg.errperc[-2])
            recordict['min15ppup'] = recordict['min15ppup'] + min15ppup
            recordict['min15ppdo'] = recordict['min15ppdo'] + min15ppdo
            if args.iscap:
                recordict['min15pup'] = recordict['min15pup'] + output[:, 1, 0].cpu().tolist()
                recordict['min15pdo'] = recordict['min15pdo'] + output[:, 1, 2].cpu().tolist()
            else:
                recordict['min15pup'] = recordict['min15pup'] + output[:, 3].cpu().tolist()
                recordict['min15pdo'] = recordict['min15pdo'] + output[:, 5].cpu().tolist()
            min30t = target[:, 2].cpu().tolist()
            min30p = output[:, 2, 1].cpu().tolist() if args.iscap else output[:, 7].cpu().tolist()
            recordict['min30t'] = recordict['min30t'] + min30t
            recordict['min30p'] = recordict['min30p'] + min30p
            # min30up, min30do = fit_interval(x=[i for i in range(len(min30t))], y=min30t)
            min30up, min30do = stats.t.interval(0.9, len(min30t)-1, loc=min30t, scale=stats.sem(min30t))
            min30up, min30do = min30up.tolist(), min30do.tolist()
            recordict['min30tup'] = recordict['min30tup'] + min30up
            recordict['min30tdo'] = recordict['min30tdo'] + min30do
            min30ppup, min30ppdo = cap_interval(x=min30t, inv=cfg.errperc[-3])
            recordict['min30ppup'] = recordict['min30ppup'] + min30ppup
            recordict['min30ppdo'] = recordict['min30ppdo'] + min30ppdo
            if args.iscap:
                recordict['min30pup'] = recordict['min30pup'] + output[:, 2, 0].cpu().tolist()
                recordict['min30pdo'] = recordict['min30pdo'] + output[:, 2, 2].cpu().tolist()
            else:
                recordict['min30pup'] = recordict['min30pup'] + output[:, 6].cpu().tolist()
                recordict['min30pdo'] = recordict['min30pdo'] + output[:, 8].cpu().tolist()
            h1t = target[:, 3].cpu().tolist()
            h1p = output[:, 3, 1].cpu().tolist() if args.iscap else output[:, 10].cpu().tolist()
            recordict['h1t'] = recordict['h1t'] + h1t
            recordict['h1p'] = recordict['h1p'] + h1p
            # h1up, h1do = fit_interval(x=[i for i in range(len(h1t))], y=h1t)
            h1up, h1do = stats.t.interval(0.9, len(h1t)-1, loc=h1t, scale=stats.sem(h1t))
            h1up, h1do = h1up.tolist(), h1do.tolist()
            recordict['h1tup'] = recordict['h1tup'] + h1up
            recordict['h1tdo'] = recordict['h1tdo'] + h1do
            h1ppup, h1ppdo = cap_interval(x=h1t, inv=cfg.errperc[-4])
            recordict['h1ppup'] = recordict['h1ppup'] + h1ppup
            recordict['h1ppdo'] = recordict['h1ppdo'] + h1ppdo
            if args.iscap:
                recordict['h1pup'] = recordict['h1pup'] + output[:, 3, 0].cpu().tolist()
                recordict['h1pdo'] = recordict['h1pdo'] + output[:, 3, 2].cpu().tolist()
            else:
                recordict['h1pup'] = recordict['h1pup'] + output[:, 9].cpu().tolist()
                recordict['h1pdo'] = recordict['h1pdo'] + output[:, 11].cpu().tolist()
            h6t = target[:, 4].cpu().tolist()
            h6p = output[:, 4, 1].cpu().tolist() if args.iscap else output[:, 13].cpu().tolist()
            recordict['h6t'] = recordict['h6t'] + h6t
            recordict['h6p'] = recordict['h6p'] + h6p
            # h6up, h6do = fit_interval(x=[i for i in range(len(h6t))], y=h6t)
            h6up, h6do = stats.t.interval(0.9, len(h6t)-1, loc=h6t, scale=stats.sem(h6t))
            h6up, h6do = h6up.tolist(), h6do.tolist()
            recordict['h6tup'] = recordict['h6tup'] + h6up
            recordict['h6tdo'] = recordict['h6tdo'] + h6do
            h6ppup, h6ppdo = cap_interval(x=h6t, inv=cfg.errperc[-5])
            recordict['h6ppup'] = recordict['h6ppup'] + h6ppup
            recordict['h6ppdo'] = recordict['h6ppdo'] + h6ppdo
            if args.iscap:
                recordict['h6pup'] = recordict['h6pup'] + output[:, 4, 0].cpu().tolist()
                recordict['h6pdo'] = recordict['h6pdo'] + output[:, 4, 2].cpu().tolist()
            else:
                recordict['h6pup'] = recordict['h6pup'] + output[:, 12].cpu().tolist()
                recordict['h6pdo'] = recordict['h6pdo'] + output[:, 14].cpu().tolist()
            d1t = target[:, 5].cpu().tolist()
            d1p = output[:, 5, 1].cpu().tolist() if args.iscap else output[:, 16].cpu().tolist()
            recordict['d1t'] = recordict['d1t'] + d1t
            recordict['d1p'] = recordict['d1p'] + d1p
            # d1up, d1do = fit_interval(x=[i for i in range(len(d1t))], y=d1t)
            d1up, d1do = stats.t.interval(0.9, len(d1t)-1, loc=d1t, scale=stats.sem(d1t))
            d1up, d1do = d1up.tolist(), d1do.tolist()
            recordict['d1tup'] = recordict['d1tup'] + d1up
            recordict['d1tdo'] = recordict['d1tdo'] + d1do
            d1ppup, d1ppdo = cap_interval(x=d1t, inv=cfg.errperc[-5])
            recordict['d1ppup'] = recordict['d1ppup'] + d1ppup
            recordict['d1ppdo'] = recordict['d1ppdo'] + d1ppdo
            if args.iscap:
                recordict['d1pup'] = recordict['d1pup'] + output[:, 5, 0].cpu().tolist()
                recordict['d1pdo'] = recordict['d1pdo'] + output[:, 5, 2].cpu().tolist()
            else:
                recordict['d1pup'] = recordict['d1pup'] + output[:, 15].cpu().tolist()
                recordict['d1pdo'] = recordict['d1pdo'] + output[:, 17].cpu().tolist()
        recordict['min5r'] = recordict['min5r'] + [ap5min_R_meter.avg.cpu().item()]
        recordict['min5rmse'] = recordict['min5rmse'] + [ap5min_RMSE_meter.avg.cpu().item()]
        recordict['min5mae'] = recordict['min5mae'] + [ap5min_MAE_meter.avg.cpu().item()]
        recordict['min5smape'] = recordict['min5smape'] + [ap5min_SMAPE_meter.avg.cpu().item()]
        recordict['min15r'] = recordict['min15r'] + [ap15min_R_meter.avg.cpu().item()]
        recordict['min15rmse'] = recordict['min15rmse'] + [ap15min_RMSE_meter.avg.cpu().item()]
        recordict['min15mae'] = recordict['min15mae'] + [ap15min_MAE_meter.avg.cpu().item()]
        recordict['min15smape'] = recordict['min15smape'] + [ap15min_SMAPE_meter.avg.cpu().item()]
        recordict['min30r'] = recordict['min30r'] + [ap30min_R_meter.avg.cpu().item()]
        recordict['min30rmse'] = recordict['min30rmse'] + [ap30min_RMSE_meter.avg.cpu().item()]
        recordict['min30mae'] = recordict['min30mae'] + [ap30min_MAE_meter.avg.cpu().item()]
        recordict['min30smape'] = recordict['min30smape'] + [ap30min_SMAPE_meter.avg.cpu().item()]
        recordict['h1r'] = recordict['h1r'] + [ap1h_R_meter.avg.cpu().item()]
        recordict['h1rmse'] = recordict['h1rmse'] + [ap1h_RMSE_meter.avg.cpu().item()]
        recordict['h1mae'] = recordict['h1mae'] + [ap1h_MAE_meter.avg.cpu().item()]
        recordict['h1smape'] = recordict['h1smape'] + [ap1h_SMAPE_meter.avg.cpu().item()]
        recordict['h6r'] = recordict['h6r'] + [ap6h_R_meter.avg.cpu().item()]
        recordict['h6rmse'] = recordict['h6rmse'] + [ap6h_RMSE_meter.avg.cpu().item()]
        recordict['h6mae'] = recordict['h6mae'] + [ap6h_MAE_meter.avg.cpu().item()]
        recordict['h6smape'] = recordict['h6smape'] + [ap6h_SMAPE_meter.avg.cpu().item()]
        recordict['d1r'] = recordict['d1r'] + [ap1d_R_meter.avg.cpu().item()]
        recordict['d1rmse'] = recordict['d1rmse'] + [ap1d_RMSE_meter.avg.cpu().item()]
        recordict['d1mae'] = recordict['d1mae'] + [ap1d_MAE_meter.avg.cpu().item()]
        recordict['d1smape'] = recordict['d1smape'] + [ap1d_SMAPE_meter.avg.cpu().item()]
        recordfs = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in recordict.items()]))
        pathcsv = os.path.join(norpath, cfg.dataname + '.csv')
        recordfs.to_csv(pathcsv, index=False)

def main(cfg, args, path_model):
    ##################################
    # Loading
    ##################################
    seed = args.seed + get_rank()
    setup_seed(seed)
    _, validate_loader = dataloader(dataroot=cfg.data_path, datacat=cfg.datacat, train_batch_size=cfg.train.batch_size, \
                                    test_batch_size=cfg.test.batch_size, workers=cfg.workers, dist=False, num_tasks=get_world_size(), \
                                    global_rank=get_rank(), image_size=cfg.train.imgsize)
    model = create_net(args)
    model.to(torch.device(args.device))
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

    ##################################
    # Start Validate
    ##################################
    test(cfg, args, model, validate_loader, path_model)


if __name__ == '__main__':
    ##################################
    # model, dataset test_batch=288
    ##################################
    cfg, args = make_cfg_args()
    weight_path = os.path.join('/home/liuyunfei/PycharmProjects/powersys/doc/result', cfg.datacat, cfg.dataname, cfg.arch, cfg.heads, cfg.run + '_0', 'model_best.pth.tar')
    main(cfg, args, weight_path)