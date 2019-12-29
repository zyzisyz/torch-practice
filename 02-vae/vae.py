#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: vae.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sun Dec 29 12:16:22 2019
# ************************************************************************/

from loss import *
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image


# --- defines the model and the optimizer --- #
class VAE(nn.Module):
	def __init__(self, z_dim):
		super().__init__()
		self.z_dim = z_dim
		# for encoder
		self.fc1 = nn.Linear(784, 500)
		self.fc21 = nn.Linear(500, self.z_dim)  # fc21 for mean of Z
		self.fc22 = nn.Linear(500, self.z_dim)  # fc22 for log variance of Z

		# for decoder
		self.fc3 = nn.Linear(self.z_dim, 500)
		self.fc4 = nn.Linear(500, 784)

	def encode(self, x):
		h1 = F.relu(self.fc1(x))
		mu = self.fc21(h1)
		# I guess the reason for using logvar instead of std or var is that
		# the output of fc22 can be negative value (std and var should be positive)
		logvar = self.fc22(h1)
		return mu, logvar

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.rand_like(std)
		return mu + eps*std

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h3))

	def forward(self, x):
		# x: [batch size, 1, 28,28] -> x: [batch size, 784]
		x = x.view(-1, 784)
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar


# --- train and test --- #
def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	train_loss = 0
	for batch_idx, (data, label) in enumerate(train_loader):
		# data: [batch size, 1, 28, 28]
		# label: [batch size] -> we don't use
		optimizer.zero_grad()
		data = data.to(device)
		recon_data, mu, logvar = model(data)
		loss = get_loss(recon_data, data, mu, logvar)
		loss.backward()
		cur_loss = loss.item()
		train_loss += cur_loss
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100.*batch_idx / len(train_loader),
				cur_loss/len(data)))
	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))


def test(args, model, device, train_loader, optimizer, epoch):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for batch_idx, (data, label) in enumerate(test_loader):
			data = data.to(device)
			recon_data, mu, logvar = model(data)
			cur_loss = get_loss(recon_data, data, mu, logvar).item()
			test_loss += cur_loss
			if batch_idx == 0:
				# saves 8 samples of the first batch as an image file to compare input images and reconstructed images
				num_samples = min(args.batch_size, 8)
				comparison = torch.cat(
						[data[:num_samples], recon_data.view(args.batch_size, 1, 28, 28)[:num_samples]]).cpu()
				save_generated_img(
						comparison, 'reconstruction', epoch, num_samples)
	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))

