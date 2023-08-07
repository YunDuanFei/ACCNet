import torch
import os
import re
from pathlib import Path
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image


__imagenet_mean_std = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class KeyError(Exception):
	def __init__(self, msg):
		self.msg = msg

	def __str__(self):
		return self.msg

def train_preproccess(image_size, normalize=__imagenet_mean_std):
	train_transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(**normalize),
		])
	return train_transforms

def test_preproccess(image_size, normalize=__imagenet_mean_std):
	test_transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(**normalize),
		])
	return test_transforms

def get_transforms(mode, image_size):
	if mode == 'train':
		train_transforms = train_preproccess(image_size=image_size)
		return train_transforms
	elif mode == 'val':
		test_transforms = test_preproccess(image_size=image_size)
		return test_transforms
	else:
		raise KeyError('mode must be train or val')

class Datasets(data.Dataset):
	def __init__(self, path, datas, labels, transform=None):
		self.path = path
		self.datas = datas
		self.labels = labels
		self.transform = transform

	def __getitem__(self, index):
		img_name = self.datas[index]
		label = self.labels[index]
		img = Image.open(os.path.join(self.path, img_name)).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		return img, label

	def __len__(self):
		return len(self.datas)

def loadimglabel(path, datacat):
	path = Path(path)
	dname = path.stem
	# load train imgs and labels
	train_txt = path / ('train' + '_' + dname + '.txt')
	with open(train_txt, 'r') as f:
		train_img_paths, train_labels = [], []
		for line in f:
			line = line.strip()
			if line.startswith('name:'):
				info = line.split('|')
				img_nam = 'train/' + str(info[0].split(':')[1].strip())
				train_img_paths.append(img_nam)
				d1ap_dat = float(info[1].split(':')[1].strip())
				h6ap_dat = float(info[2].split(':')[1].strip())
				h1ap_dat = float(info[3].split(':')[1].strip())
				min30ap_dat = float(info[4].split(':')[1].strip())
				min15ap_dat = float(info[5].split(':')[1].strip())
				min5ap_dat = float(info[6].split(':')[1].strip())
				train_labels.append([d1ap_dat, h1ap_dat, h6ap_dat, min30ap_dat, min15ap_dat, min5ap_dat])
	# load test imgs and labels
	test_txt = path / ('test' + '_' + dname + '.txt')
	with open(test_txt, 'r') as f:
		test_img_paths, test_labels = [], []
		for line in f:
			line = line.strip()
			if line.startswith('name:'):
				info = line.split('|')
				img_nam = 'test/' + (info[0].split(':')[1].strip())
				test_img_paths.append(img_nam)
				d1ap_dat = float(info[1].split(':')[1].strip())
				h6ap_dat = float(info[2].split(':')[1].strip())
				h1ap_dat = float(info[3].split(':')[1].strip())
				min30ap_dat = float(info[4].split(':')[1].strip())
				min15ap_dat = float(info[5].split(':')[1].strip())
				min5ap_dat = float(info[6].split(':')[1].strip())
				test_labels.append([d1ap_dat, h1ap_dat, h6ap_dat, min30ap_dat, min15ap_dat, min5ap_dat])
	return train_img_paths, train_labels, test_img_paths, test_labels

def dataloader(dataroot, datacat, train_batch_size, test_batch_size, dist, num_tasks, global_rank, workers=4, image_size=72):
	img_train, label_train, img_test, label_test = loadimglabel(path=dataroot, datacat=datacat)
	train_dataset = Datasets(path=dataroot, datas=img_train, labels=label_train, transform=get_transforms(mode='train', image_size=image_size))
	val_dataset = Datasets(path=dataroot, datas=img_test, labels=label_test, transform=get_transforms(mode='val', image_size=image_size))
	if dist:
		train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
	else:
		train_sampler = torch.utils.data.RandomSampler(train_dataset)
	train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, num_workers=workers, \
		pin_memory=True, drop_last=True,)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=workers, \
		pin_memory=True, drop_last=False)
	return train_loader, val_loader