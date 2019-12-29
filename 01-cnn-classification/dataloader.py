#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: dataloader.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sun Dec 29 11:46:57 2019
# ************************************************************************/


from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# load the train and test fashion mnist dataset
def dataloader(batch_size):

	kwargs = {'num_workers': 1, 'pin_memory': True}

	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])

	mnist_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True, transform=transform), 
	train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, **kwargs)

	mnist_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True, transform=transform), 
	test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True,**kwargs)

	return train_loader, test_loader

