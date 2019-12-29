#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: train.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sun Dec 29 10:35:46 2019
# ************************************************************************/

from cnn import *
from dataloader import *
import argparse


def main():

	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch Fashion-MNIST CNN-test')

	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')

	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')

	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')

	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')

	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')

	parser.add_argument('--cuda', action='store_true', default=True,
						help='enable CUDA training')

	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')

	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')

	args = parser.parse_args()

	# check and print training decive
	use_cuda = args.cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print("device: ",device)

	torch.manual_seed(args.seed)



	# load the fashion mnist dataset
	train_loader, test_loader = dataloader(args.epochs)

	model = Net().to(device)
	print(model)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

	for epoch in range(args.epochs):
		train(args, model, device, train_loader, optimizer, epoch)
		test(args, model, device, test_loader)
		if (args.save_model):
			if os.path.exists("model") == False:
				os.mkdir("model")
			torch.save(model.state_dict(),"model/cnn_{0}.pt".format(epoch))

	if (args.save_model):
		if os.path.exists("model") == False:
			os.mkdir("model")
		torch.save(model.state_dict(),"model/cnn_{0}.pt".format(epoch))


if __name__ == '__main__':
	main()
