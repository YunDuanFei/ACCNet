import os
import torch
import logging
import warnings
import torchvision
import tempfile
import pathlib
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.distributed as dist
from configs import make_cfg_args
from datasets import dataloader
from networks import create_net
from utils import *
import warnings
warnings.filterwarnings('ignore')


def main(cfg, args, best_prec1):
	##################################
	# seed and dist init
	##################################
	init_distributed_mode(args)
	device = torch.device(args.device)
	seed = args.seed + get_rank()
	setup_seed(seed)
	torch.backends.cudnn.benchmark = True

	##################################
	# Logging setting
	##################################
	os.makedirs(cfg.record_dir, exist_ok=True)
	logging.basicConfig(
						filename=os.path.join(cfg.record_dir, cfg.run + '_' + str(args.seed) + '.log'),
						filemode='w',
						format='%(asctime)s: %(message)s',
						level=logging.INFO)
	warnings.filterwarnings("ignore")

	##################################
	# Load sampler dataset
	##################################
	train_loader, validate_loader = dataloader(dataroot=cfg.data_path, datacat=cfg.datacat, train_batch_size=cfg.train.batch_size, \
		test_batch_size=cfg.test.batch_size, workers=cfg.workers, dist=args.distributed, num_tasks=get_world_size(), \
		global_rank=get_rank(), image_size=cfg.train.imgsize)

	##################################
	# Load model
	##################################
	model = create_net(args)
	flops, params = get_model_complexity_info(model, (cfg.train.imgsize, cfg.train.imgsize), as_strings=False, print_per_layer_stat=False)
	model.to(device)
	if args.distributed:
		checkpoint_path = os.path.join(tempfile.gettempdir(), 'initial_weights.pt')
		if args.rank == 0:
			torch.save(model.state_dict(), checkpoint_path)
		dist.barrier()
		model.load_state_dict(torch.load(checkpoint_path, map_location=device))
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

	##################################
	# tensorboard setting
	##################################
	writer_path = pathlib.Path(cfg.tensor_Board, cfg.datacat, cfg.dataname, cfg.arch, cfg.heads, cfg.run + '_' + str(args.seed))
	writer_path.mkdir(parents=True, exist_ok=True)
	Writer = SummaryWriter(log_dir=writer_path)

	##################################
	# Load optimizer, scheduler, loss
	##################################
	criterion = CompleLoss(percs=cfg.errperc, iscap=args.iscap).cuda()
	optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr*float(cfg.train.batch_size*args.world_size)/256., \
		momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
	scheduler = CosineAnnealingLR(optimizer, T_max=(cfg.train.epoch-args.warm)*len(train_loader), warmup='linear', \
		warmup_iters=args.warm*len(train_loader), eta_min=1e-8)

	##################################
	# Logging title
	##################################
	logging.info('-------------------------------config--------------------------------')
	logging.info(cfg)
	logging.info('-------------------------network information-------------------------')
	logging.info('Flops:  {:.3f}GMac  Params: {:.2f}M'.format(flops / 1e9, params / 1e6))
	logging.info('Network weights save to {}'.format(cfg.record_dir))
	logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.format(cfg.train.epoch, \
		cfg.train.batch_size, cfg.imagenet_train_size, cfg.imagenet_test_size))
	logging.info('-------------------------------model--------------------------------')
	logging.info(model)
	logging.info('--------------------------training progress--------------------------')

	##################################
	# Start trainging
	##################################
	for epoch in range(0, cfg.train.epoch):
		if args.distributed:
			train_loader.sampler.set_epoch(epoch)
		train_matrics = train(model, train_loader, epoch, optimizer, criterion, device, scheduler, cfg, args)
		logging.info('Training epoch = {}'.format(epoch))
		logging.info('	Loss: {:.4f}. Train epoch time: {:.4f}.'.format(train_matrics['loss'], train_matrics['epoch_t']))
		logging.info('	ap1d_R: {:.4f}. ap1d_RMSE: {:.4f}. ap1d_MAE: {:.4f}. ap1d_SMAPE: {:.4f}. '.format(train_matrics['ap1d_R'], train_matrics['ap1d_RMSE'], train_matrics['ap1d_MAE'], train_matrics['ap1d_SMAPE']))
		logging.info('	ap6h_R: {:.4f}. ap6h_RMSE: {:.4f}. ap6h_MAE: {:.4f}. ap6h_SMAPE: {:.4f}.'.format(train_matrics['ap6h_R'], train_matrics['ap6h_RMSE'], train_matrics['ap6h_MAE'], train_matrics['ap6h_SMAPE']))
		logging.info('	ap1h_R: {:.4f}. ap1h_RMSE: {:.4f}. ap1h_MAE: {:.4f}. ap1h_SMAPE: {:.4f}.'.format(train_matrics['ap1h_R'], train_matrics['ap1h_RMSE'], train_matrics['ap1h_MAE'], train_matrics['ap1h_SMAPE']))
		logging.info('	ap30min_R: {:.4f}. ap30min_RMSE: {:.4f}. ap30min_MAE: {:.4f}. ap30min_SMAPE: {:.4f}.'.format(train_matrics['ap30min_R'], train_matrics['ap30min_RMSE'], train_matrics['ap30min_MAE'], train_matrics['ap30min_SMAPE']))
		logging.info('	ap15min_R: {:.4f}. ap15min_RMSE: {:.4f}. ap15min_MAE: {:.4f}. ap15min_SMAPE: {:.4f}.'.format(train_matrics['ap15min_R'], train_matrics['ap15min_RMSE'], train_matrics['ap15min_MAE'], train_matrics['ap15min_SMAPE']))
		logging.info('	ap5min_R: {:.4f}. ap5min_RMSE: {:.4f}. ap5min_MAE: {:.4f}. ap5min_SMAPE: {:.4f}.'.format(train_matrics['ap5min_R'], train_matrics['ap5min_RMSE'], train_matrics['ap5min_MAE'], train_matrics['ap5min_SMAPE']))

		test_matrics = test(model, validate_loader, criterion, device, cfg, args)
		logging.info('Testing epoch = {}'.format(epoch))
		logging.info('	Loss: {:.4f}. Train epoch time: {:.4f}.'.format(test_matrics['loss'], test_matrics['epoch_t']))
		logging.info('	ap1d_R: {:.4f}. ap1d_RMSE: {:.4f}. ap1d_MAE: {:.4f}. ap1d_SMAPE: {:.4f}. '.format(test_matrics['ap1d_R'], test_matrics['ap1d_RMSE'], test_matrics['ap1d_MAE'], test_matrics['ap1d_SMAPE']))
		logging.info('	ap6h_R: {:.4f}. ap6h_RMSE: {:.4f}. ap6h_MAE: {:.4f}. ap6h_SMAPE: {:.4f}.'.format(test_matrics['ap6h_R'], test_matrics['ap6h_RMSE'], test_matrics['ap6h_MAE'], test_matrics['ap6h_SMAPE']))
		logging.info('	ap1h_R: {:.4f}. ap1h_RMSE: {:.4f}. ap1h_MAE: {:.4f}. ap1h_SMAPE: {:.4f}.'.format(test_matrics['ap1h_R'], test_matrics['ap1h_RMSE'], test_matrics['ap1h_MAE'], test_matrics['ap1h_SMAPE']))
		logging.info('	ap30min_R: {:.4f}. ap30min_RMSE: {:.4f}. ap30min_MAE: {:.4f}. ap30min_SMAPE: {:.4f}.'.format(test_matrics['ap30min_R'], test_matrics['ap30min_RMSE'], test_matrics['ap30min_MAE'], test_matrics['ap30min_SMAPE']))
		logging.info('	ap15min_R: {:.4f}. ap15min_RMSE: {:.4f}. ap15min_MAE: {:.4f}. ap15min_SMAPE: {:.4f}.'.format(test_matrics['ap15min_R'], test_matrics['ap15min_RMSE'], test_matrics['ap15min_MAE'], test_matrics['ap15min_SMAPE']))
		logging.info('	ap5min_R: {:.4f}. ap5min_RMSE: {:.4f}. ap5min_MAE: {:.4f}. ap5min_SMAPE: {:.4f}.'.format(test_matrics['ap5min_R'], test_matrics['ap5min_RMSE'], test_matrics['ap5min_MAE'], test_matrics['ap5min_SMAPE']))
		logging.info('Epoch {:03d}, Learning Rate {:g}\n'.format(epoch, optimizer.param_groups[0]['lr']))
		Writer.add_scalar('train_loss',                     train_matrics['loss'], epoch)
		Writer.add_scalar('train_ap1d_R',                 train_matrics['ap1d_R'], epoch)
		Writer.add_scalar('train_ap1d_RMSE',           train_matrics['ap1d_RMSE'], epoch)
		Writer.add_scalar('train_ap1d_MAE',             train_matrics['ap1d_MAE'], epoch)
		Writer.add_scalar('train_ap1d_SMAPE',         train_matrics['ap1d_SMAPE'], epoch)
		Writer.add_scalar('train_ap6h_R',                 train_matrics['ap6h_R'], epoch)
		Writer.add_scalar('train_ap6h_RMSE',           train_matrics['ap6h_RMSE'], epoch)
		Writer.add_scalar('train_ap6h_MAE',             train_matrics['ap6h_MAE'], epoch)
		Writer.add_scalar('train_ap6h_SMAPE',         train_matrics['ap6h_SMAPE'], epoch)
		Writer.add_scalar('train_ap1h_R',                 train_matrics['ap1h_R'], epoch)
		Writer.add_scalar('train_ap1h_RMSE',           train_matrics['ap1h_RMSE'], epoch)
		Writer.add_scalar('train_ap1h_MAE',             train_matrics['ap1h_MAE'], epoch)
		Writer.add_scalar('train_ap1h_SMAPE',         train_matrics['ap1h_SMAPE'], epoch)
		Writer.add_scalar('train_ap30min_R',           train_matrics['ap30min_R'], epoch)
		Writer.add_scalar('train_ap30min_RMSE',     train_matrics['ap30min_RMSE'], epoch)
		Writer.add_scalar('train_ap30min_MAE',       train_matrics['ap30min_MAE'], epoch)
		Writer.add_scalar('train_ap30min_SMAPE',   train_matrics['ap30min_SMAPE'], epoch)
		Writer.add_scalar('train_ap15min_R',           train_matrics['ap15min_R'], epoch)
		Writer.add_scalar('train_ap15min_RMSE',     train_matrics['ap15min_RMSE'], epoch)
		Writer.add_scalar('train_ap15min_MAE',       train_matrics['ap15min_MAE'], epoch)
		Writer.add_scalar('train_ap15min_SMAPE',   train_matrics['ap15min_SMAPE'], epoch)
		Writer.add_scalar('train_ap5min_R',             train_matrics['ap5min_R'], epoch)
		Writer.add_scalar('train_ap5min_RMSE',       train_matrics['ap5min_RMSE'], epoch)
		Writer.add_scalar('train_ap5min_MAE',         train_matrics['ap5min_MAE'], epoch)
		Writer.add_scalar('train_ap5min_SMAPE',     train_matrics['ap5min_SMAPE'], epoch)
		Writer.add_scalar('test_loss',                     test_matrics['loss'], epoch)
		Writer.add_scalar('test_ap1d_R',                 test_matrics['ap1d_R'], epoch)
		Writer.add_scalar('test_ap1d_RMSE',           test_matrics['ap1d_RMSE'], epoch)
		Writer.add_scalar('test_ap1d_MAE',             test_matrics['ap1d_MAE'], epoch)
		Writer.add_scalar('test_ap1d_SMAPE',         test_matrics['ap1d_SMAPE'], epoch)
		Writer.add_scalar('test_ap6h_R',                 test_matrics['ap6h_R'], epoch)
		Writer.add_scalar('test_ap6h_RMSE',           test_matrics['ap6h_RMSE'], epoch)
		Writer.add_scalar('test_ap6h_MAE',             test_matrics['ap6h_MAE'], epoch)
		Writer.add_scalar('test_ap6h_SMAPE',         test_matrics['ap6h_SMAPE'], epoch)
		Writer.add_scalar('test_ap1h_R',                 test_matrics['ap1h_R'], epoch)
		Writer.add_scalar('test_ap1h_RMSE',           test_matrics['ap1h_RMSE'], epoch)
		Writer.add_scalar('test_ap1h_MAE',             test_matrics['ap1h_MAE'], epoch)
		Writer.add_scalar('test_ap1h_SMAPE',         test_matrics['ap1h_SMAPE'], epoch)
		Writer.add_scalar('test_ap30min_R',           test_matrics['ap30min_R'], epoch)
		Writer.add_scalar('test_ap30min_RMSE',     test_matrics['ap30min_RMSE'], epoch)
		Writer.add_scalar('test_ap30min_MAE',       test_matrics['ap30min_MAE'], epoch)
		Writer.add_scalar('test_ap30min_SMAPE',   test_matrics['ap30min_SMAPE'], epoch)
		Writer.add_scalar('test_ap15min_R',           test_matrics['ap15min_R'], epoch)
		Writer.add_scalar('test_ap15min_RMSE',     test_matrics['ap15min_RMSE'], epoch)
		Writer.add_scalar('test_ap15min_MAE',       test_matrics['ap15min_MAE'], epoch)
		Writer.add_scalar('test_ap15min_SMAPE',   test_matrics['ap15min_SMAPE'], epoch)
		Writer.add_scalar('test_ap5min_R',             test_matrics['ap5min_R'], epoch)
		Writer.add_scalar('test_ap5min_RMSE',       test_matrics['ap5min_RMSE'], epoch)
		Writer.add_scalar('test_ap5min_MAE',         test_matrics['ap5min_MAE'], epoch)
		Writer.add_scalar('test_ap5min_SMAPE',     test_matrics['ap5min_SMAPE'], epoch)

		##################################
		# Check point
		#################################
		is_best = test_matrics['ap5min_R'] > best_prec1
		if is_best or cfg.train.epoch <= 1:
			ap5min_R, ap5min_RMSE, ap5min_MAE, ap5min_SMAPE     = test_matrics['ap5min_R'], test_matrics['ap5min_RMSE'], test_matrics['ap5min_MAE'], test_matrics['ap5min_SMAPE']
			ap15min_R, ap15min_RMSE, ap15min_MAE, ap15min_SMAPE = test_matrics['ap15min_R'], test_matrics['ap15min_RMSE'], test_matrics['ap15min_MAE'], test_matrics['ap15min_SMAPE']
			ap30min_R, ap30min_RMSE, ap30min_MAE, ap30min_SMAPE = test_matrics['ap30min_R'], test_matrics['ap30min_RMSE'], test_matrics['ap30min_MAE'], test_matrics['ap30min_SMAPE']
			ap1h_R, ap1h_RMSE, ap1h_MAE, ap1h_SMAPE = test_matrics['ap1h_R'], test_matrics['ap1h_RMSE'], test_matrics['ap1h_MAE'], test_matrics['ap1h_SMAPE']
			ap6h_R, ap6h_RMSE, ap6h_MAE, ap6h_SMAPE = test_matrics['ap6h_R'], test_matrics['ap6h_RMSE'], test_matrics['ap6h_MAE'], test_matrics['ap6h_SMAPE']
			ap1d_R, ap1d_RMSE, ap1d_MAE, ap1d_SMAPE = test_matrics['ap1d_R'], test_matrics['ap1d_RMSE'], test_matrics['ap1d_MAE'], test_matrics['ap1d_SMAPE']

		best_prec1 = max(test_matrics['ap5min_R'], best_prec1)
		if get_rank() == 0:
			save_checkpoint({
				'epoch': epoch+1,
				'arch': args.arch,
				'state_dict': model.module.state_dict(),
				'best_prec1': best_prec1,
				'optimizer' : optimizer.state_dict(),
			}, is_best, cfg, args)
	if args.rank == 0:
		if os.path.exists(checkpoint_path) is True:
			os.remove(checkpoint_path)

	##################################
	# End trainging
	#################################
	logging.info('================= END =================')
	logging.info(f'ap5min_R  {ap5min_R :.4f} | ap5min_RMSE  {ap5min_RMSE :.4f} | ap5min_MAE  {ap5min_MAE :.4f} | ap5min_SMAPE {ap5min_SMAPE :.4f}')
	logging.info(f'ap15min_R {ap15min_R :.4f} | ap15min_RMSE {ap15min_RMSE :.4f} | ap15min_MAE {ap15min_MAE :.4f} | ap15min_SMAPE {ap15min_SMAPE :.4f}')
	logging.info(f'ap30min_R {ap30min_R :.4f} | ap30min_RMSE {ap30min_RMSE :.4f} | ap30min_MAE {ap30min_MAE :.4f} | ap30min_SMAPE {ap30min_SMAPE :.4f}')
	logging.info(f'ap1h_R {ap1h_R :.4f} | ap1h_RMSE {ap1h_RMSE :.4f} | ap1h_MAE {ap1h_MAE :.4f} | ap1h_SMAPE {ap1h_SMAPE :.4f}')
	logging.info(f'ap6h_R {ap6h_R :.4f} | ap6h_RMSE {ap6h_RMSE :.4f} | ap6h_MAE {ap6h_MAE :.4f} | ap6h_SMAPE {ap6h_SMAPE :.4f}')
	logging.info(f'ap1d_R {ap1d_R :.4f} | ap1d_RMSE {ap1d_RMSE :.4f} | ap1d_MAE {ap1d_MAE :.4f} | ap1d_SMAPE {ap1d_SMAPE :.4f}')
	Writer.close()
	dist.destroy_process_group()


if __name__ == '__main__':
	cfg, args = make_cfg_args()
	main(cfg, args, best_prec1=0)