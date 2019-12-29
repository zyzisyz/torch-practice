#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: train.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sun Dec 29 12:30:53 2019
# ************************************************************************/

from vae import *
from dataloader import *
import argparse, os, torch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

def main():
	parser = argparse.ArgumentParser(description="PyTorch implementation of VAE for fashion-MNIST")
	parser.add_argument('--batch-size', type=int, default=128,
			help='batch size for training (default: 128)')
	parser.add_argument('--epochs', type=int, default=20,
			help='number of epochs to train (default: 20)')
	parser.add_argument('--z_dim', type=int, default=2,
			help='dimension of hidden variable Z (default: 2)')
	parser.add_argument('--log-interval', type=int, default=100,
			help='interval between logs about training status (default: 100)')
	parser.add_argument('--lr', type=float, default=1e-3,
			help='learning rate for Adam optimizer (default: 1e-3)')
	parser.add_argument('--prr', type=bool, default=True,
			help='Boolean for plot-reproduce-result (default: True')
	parser.add_argument('--prr-z1-range', type=int, default=2,
			help='z1 range for plot-reproduce-result (default: 2)')
	parser.add_argument('--prr-z2-range', type=int, default=2,
			help='z2 range for plot-reproduce-result (default: 2)')
	parser.add_argument('--prr-z1-interval', type=int, default=0.2,
			help='interval of z1 for plot-reproduce-result (default: 0.2)')
	parser.add_argument('--prr-z2-interval', type=int, default=0.2,
			help='interval of z2 for plot-reproduce-result (default: 0.2)')
	parser.add_argument('--save-model', type=bool, default=True,
			help='Boolean for save the model (default: True')
	args = parser.parse_args()


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# load the fashion mnist dataset
	train_loader, test_loader = dataloader(args.epochs)

	model = VAE(args.z_dim).to(device)
	print(model)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	for epoch in range(args.epochs):
		train(args, model, device, train_loader, optimizer, epoch)
		if (args.save_model):
			if os.path.exists("model") == False:
				os.mkdir("model")
			torch.save(model.state_dict(),"model/vae_{0}.pt".format(epoch))

	if (args.save_model):
		if os.path.exists("model") == False:
			os.mkdir("model")
		torch.save(model.state_dict(),"model/vae_{0}.pt".format(epoch))


if __name__=="__main__":
	main()

