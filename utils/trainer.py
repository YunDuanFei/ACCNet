import torch
import time
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
from torch.cuda.amp import GradScaler,autocast
from .util import ProgressMeter, reduce_value, get_rank

def train(model, train_loader, epoch, optimizer, criterion, device, scheduler, cfg, args):
    start = time.time()
    model.train()
    batch_time = AverageMeter('Batch Time', ':6.3f')
    epoch_time = AverageMeter('Epoch Time', ':6.3f')
    data_time  = AverageMeter('Data Time', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')
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
    scaler = GradScaler()
    end = time.time()

    if get_rank():
        train_loader = tqdm(train_loader, leave=False, desc='training')

    for batch_idx, (data, target) in enumerate(train_loader):
        scheduler.step()
        data_time.update(time.time() - end)
        data, target = data.to(device=device, non_blocking=True), torch.stack(target, dim=1).to(device=device, non_blocking=True).float()
        optimizer.zero_grad()
        if cfg.using_amp:
            # Using fp16 
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            if cfg.train.clip_gradient:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.train.max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Using fp32
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if cfg.train.clip_gradient:
                clip_grad_norm_(model.parameters(), cfg.train.max_norm)
            optimizer.step()
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
        if args.distributed:
            loss = reduce_value(loss.data)
            ap1d_R = reduce_value(ap1d_R)
            ap1d_RMSE = reduce_value(ap1d_RMSE)
            ap1d_MAE = reduce_value(ap1d_MAE)
            ap1d_SMAPE = reduce_value(ap1d_SMAPE)
            ap6h_R = reduce_value(ap6h_R)
            ap6h_RMSE = reduce_value(ap6h_RMSE)
            ap6h_MAE = reduce_value(ap6h_MAE)
            ap6h_SMAPE = reduce_value(ap6h_SMAPE)
            ap1h_R = reduce_value(ap1h_R)
            ap1h_RMSE = reduce_value(ap1h_RMSE)
            ap1h_MAE = reduce_value(ap1h_MAE)
            ap1h_SMAPE = reduce_value(ap1h_SMAPE)
            ap30min_R = reduce_value(ap30min_R)
            ap30min_RMSE = reduce_value(ap30min_RMSE)
            ap30min_MAE = reduce_value(ap30min_MAE)
            ap30min_SMAPE = reduce_value(ap30min_SMAPE)
            ap15min_R = reduce_value(ap15min_R)
            ap15min_RMSE = reduce_value(ap15min_RMSE)
            ap15min_MAE = reduce_value(ap15min_MAE)
            ap15min_SMAPE = reduce_value(ap15min_SMAPE)
            ap5min_R = reduce_value(ap5min_R)
            ap5min_RMSE = reduce_value(ap5min_RMSE)
            ap5min_MAE = reduce_value(ap5min_MAE)
            ap5min_SMAPE = reduce_value(ap5min_SMAPE)
        else:
            loss = loss.data
        loss_meter.update(loss.item(), data.size(0))
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

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx != 0 and batch_idx % cfg.train.log_interval == 0 and get_rank():
            tqdm.write(
                f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}({100. * batch_idx / len(train_loader):.0f}%)]. '
                f'ap1d_R: {ap1d_R_meter.avg:.4f}. '
                f'ap1d_RMSE: {ap1d_RMSE_meter.avg:.4f}. '
                f'ap1d_MAE: {ap1d_MAE_meter.avg:.4f}. '
                f'ap1d_SMAPE: {ap1d_SMAPE_meter.avg:.4f}. '
                f'ap6h_R: {ap6h_R_meter.avg:.4f}. '
                f'ap6h_RMSE: {ap6h_RMSE_meter.avg:.4f}. '
                f'ap6h_MAE: {ap6h_MAE_meter.avg:.4f}. '
                f'ap6h_SMAPE: {ap6h_SMAPE_meter.avg:.4f}. '
                f'ap1h_R: {ap1h_R_meter.avg:.4f}. '
                f'ap1h_RMSE: {ap1h_RMSE_meter.avg:.4f}. '
                f'ap1h_MAE: {ap1h_MAE_meter.avg:.4f}. '
                f'ap1h_SMAPE: {ap1h_SMAPE_meter.avg:.4f}. '
                f'ap30min_R: {ap30min_R_meter.avg:.4f}. '
                f'ap30min_RMSE: {ap30min_RMSE_meter.avg:.4f}. '
                f'ap30min_MAE: {ap30min_MAE_meter.avg:.4f}. '
                f'ap30min_SMAPE: {ap30min_SMAPE_meter.avg:.4f}. '
                f'ap15min_R: {ap15min_R_meter.avg:.4f}. '
                f'ap15min_RMSE: {ap15min_RMSE_meter.avg:.4f}. '
                f'ap15min_MAE: {ap15min_MAE_meter.avg:.4f}. '
                f'ap15min_SMAPE: {ap15min_SMAPE_meter.avg:.4f}. '
                f'ap5min_R: {ap5min_R_meter.avg:.4f}. '
                f'ap5min_RMSE: {ap5min_RMSE_meter.avg:.4f}. '
                f'ap5min_MAE: {ap5min_MAE_meter.avg:.4f}. '
                f'ap5min_SMAPE: {ap5min_SMAPE_meter.avg:.4f}. '
                f'Loss: {loss_meter.avg:.4f}. '
                f'Data time: {data_time.avg:.5f}. '
                f'Batch time: {batch_time.avg:.5f}. '
                )
    epoch_time.update(time.time() - start)
    train_matrics = dict([
        ('loss', loss_meter.avg),
        ('ap1d_R', ap1d_R_meter.avg),
        ('ap1d_RMSE', ap1d_RMSE_meter.avg),
        ('ap1d_MAE', ap1d_MAE_meter.avg),
        ('ap1d_SMAPE', ap1d_SMAPE_meter.avg),
        ('ap6h_R', ap6h_R_meter.avg),
        ('ap6h_RMSE', ap6h_RMSE_meter.avg),
        ('ap6h_MAE', ap6h_MAE_meter.avg),
        ('ap6h_SMAPE', ap6h_SMAPE_meter.avg),
        ('ap1h_R', ap1h_R_meter.avg),
        ('ap1h_RMSE', ap1h_RMSE_meter.avg),
        ('ap1h_MAE', ap1h_MAE_meter.avg),
        ('ap1h_SMAPE', ap1h_SMAPE_meter.avg),
        ('ap30min_R', ap30min_R_meter.avg),
        ('ap30min_RMSE', ap30min_RMSE_meter.avg),
        ('ap30min_MAE', ap30min_MAE_meter.avg),
        ('ap30min_SMAPE', ap30min_SMAPE_meter.avg),
        ('ap15min_R', ap15min_R_meter.avg),
        ('ap15min_RMSE', ap15min_RMSE_meter.avg),
        ('ap15min_MAE', ap15min_MAE_meter.avg),
        ('ap15min_SMAPE', ap15min_SMAPE_meter.avg),
        ('ap5min_R', ap5min_R_meter.avg),
        ('ap5min_RMSE', ap5min_RMSE_meter.avg),
        ('ap5min_MAE', ap5min_MAE_meter.avg),
        ('ap5min_SMAPE', ap5min_SMAPE_meter.avg),
        ('epoch_t', epoch_time.avg)
        ])
    return train_matrics


def test(model, val_loader, criterion, device, cfg, args):
    model.eval()
    start = time.time()
    epoch_time = AverageMeter('Epoch Time', ':6.4f')
    loss_meter = AverageMeter('Loss', ':.4e')
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
    progress = ProgressMeter(len(val_loader), [loss_meter, ap1d_R_meter, ap1d_RMSE_meter, ap1d_MAE_meter, ap1d_SMAPE_meter,  \
        ap6h_R_meter, ap6h_RMSE_meter, ap6h_MAE_meter, ap6h_SMAPE_meter, ap1h_R_meter, ap1h_RMSE_meter, ap1h_MAE_meter, ap1h_SMAPE_meter, \
        ap30min_R_meter, ap30min_RMSE_meter, ap30min_MAE_meter, ap30min_SMAPE_meter,  ap15min_R_meter, ap15min_RMSE_meter, ap15min_MAE_meter, ap15min_SMAPE_meter, \
        ap5min_R_meter, ap5min_RMSE_meter, ap5min_MAE_meter, ap5min_SMAPE_meter], prefix='Test: ')

    if get_rank():
        val_loader = tqdm(val_loader, leave=False, desc='evaluating')

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device=device, non_blocking=True), torch.stack(target, dim=1).to(device=device, non_blocking=True).float()
            output = model(data)
            loss = criterion(output, target)
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
            if args.distributed:
                loss = reduce_value(loss.data)
                ap1d_R = reduce_value(ap1d_R)
                ap1d_RMSE = reduce_value(ap1d_RMSE)
                ap1d_MAE = reduce_value(ap1d_MAE)
                ap1d_SMAPE = reduce_value(ap1d_SMAPE)
                ap6h_R = reduce_value(ap6h_R)
                ap6h_RMSE = reduce_value(ap6h_RMSE)
                ap6h_MAE = reduce_value(ap6h_MAE)
                ap6h_SMAPE = reduce_value(ap6h_SMAPE)
                ap1h_R = reduce_value(ap1h_R)
                ap1h_RMSE = reduce_value(ap1h_RMSE)
                ap1h_MAE = reduce_value(ap1h_MAE)
                ap1h_SMAPE = reduce_value(ap1h_SMAPE)
                ap30min_R = reduce_value(ap30min_R)
                ap30min_RMSE = reduce_value(ap30min_RMSE)
                ap30min_MAE = reduce_value(ap30min_MAE)
                ap30min_SMAPE = reduce_value(ap30min_SMAPE)
                ap15min_R = reduce_value(ap15min_R)
                ap15min_RMSE = reduce_value(ap15min_RMSE)
                ap15min_MAE = reduce_value(ap15min_MAE)
                ap15min_SMAPE = reduce_value(ap15min_SMAPE)
                ap5min_R = reduce_value(ap5min_R)
                ap5min_RMSE = reduce_value(ap5min_RMSE)
                ap5min_MAE = reduce_value(ap5min_MAE)
                ap5min_SMAPE = reduce_value(ap5min_SMAPE)

            else:
                loss = loss.data
            loss_meter.update(loss.item(), data.size(0))
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

            if batch_idx != 0 and  batch_idx % cfg.test.log_interval == 0 and get_rank():
                progress.display(batch_idx)

    epoch_time.update(time.time() - start)
    if get_rank():
        tqdm.write(
            f'Test set: Epoch inference time: {epoch_time.avg:.4f}. '
            f'Average loss: {loss_meter.avg:.4f}. '
            f'ap1d_R: {ap1d_R_meter.avg:.4f}. '
            f'ap1d_RMSE: {ap1d_RMSE_meter.avg:.4f}. '
            f'ap1d_MAE: {ap1d_MAE_meter.avg:.4f}. '
            f'ap1d_SMAPE: {ap1d_SMAPE_meter.avg:.4f}. '
            f'ap6h_R: {ap6h_R_meter.avg:.4f}. '
            f'ap6h_RMSE: {ap6h_RMSE_meter.avg:.4f}. '
            f'ap6h_MAE: {ap6h_MAE_meter.avg:.4f}. '
            f'ap6h_SMAPE: {ap6h_SMAPE_meter.avg:.4f}. '
            f'ap1h_R: {ap1h_R_meter.avg:.4f}. '
            f'ap1h_RMSE: {ap1h_RMSE_meter.avg:.4f}. '
            f'ap1h_MAE: {ap1h_MAE_meter.avg:.4f}. '
            f'ap1h_SMAPE: {ap1h_SMAPE_meter.avg:.4f}. '
            f'ap30min_R: {ap30min_R_meter.avg:.4f}. '
            f'ap30min_RMSE: {ap30min_RMSE_meter.avg:.4f}. '
            f'ap30min_MAE: {ap30min_MAE_meter.avg:.4f}. '
            f'ap30min_SMAPE: {ap30min_SMAPE_meter.avg:.4f}. '
            f'ap15min_R: {ap15min_R_meter.avg:.4f}. '
            f'ap15min_RMSE: {ap15min_RMSE_meter.avg:.4f}. '
            f'ap15min_MAE: {ap15min_MAE_meter.avg:.4f}. '
            f'ap15min_SMAPE: {ap15min_SMAPE_meter.avg:.4f}. '
            f'ap5min_R: {ap5min_R_meter.avg:.4f}. '
            f'a5min_RMSE: {ap5min_RMSE_meter.avg:.4f}. '
            f'ap5min_MAE: {ap5min_MAE_meter.avg:.4f}. '
            f'ap5min_SMAPE: {ap5min_SMAPE_meter.avg:.4f}. '
            )
    test_matrics = dict([
        ('loss', loss_meter.avg),
        ('ap1d_R', ap1d_R_meter.avg),
        ('ap1d_RMSE', ap1d_RMSE_meter.avg),
        ('ap1d_MAE', ap1d_MAE_meter.avg),
        ('ap1d_SMAPE', ap1d_SMAPE_meter.avg),
        ('ap6h_R', ap6h_R_meter.avg),
        ('ap6h_RMSE', ap6h_RMSE_meter.avg),
        ('ap6h_MAE', ap6h_MAE_meter.avg),
        ('ap6h_SMAPE', ap6h_SMAPE_meter.avg),
        ('ap1h_R', ap1h_R_meter.avg),
        ('ap1h_RMSE', ap1h_RMSE_meter.avg),
        ('ap1h_MAE', ap1h_MAE_meter.avg),
        ('ap1h_SMAPE', ap1h_SMAPE_meter.avg),
        ('ap30min_R', ap30min_R_meter.avg),
        ('ap30min_RMSE', ap30min_RMSE_meter.avg),
        ('ap30min_MAE', ap30min_MAE_meter.avg),
        ('ap30min_SMAPE', ap30min_SMAPE_meter.avg),
        ('ap15min_R', ap15min_R_meter.avg),
        ('ap15min_RMSE', ap15min_RMSE_meter.avg),
        ('ap15min_MAE', ap15min_MAE_meter.avg),
        ('ap15min_SMAPE', ap15min_SMAPE_meter.avg),
        ('ap5min_R', ap5min_R_meter.avg),
        ('ap5min_RMSE', ap5min_RMSE_meter.avg),
        ('ap5min_MAE', ap5min_MAE_meter.avg),
        ('ap5min_SMAPE', ap5min_SMAPE_meter.avg),
        ('epoch_t', epoch_time.avg)
        ])
    return test_matrics

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