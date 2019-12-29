#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: loss.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sun Dec 29 12:27:45 2019
# ************************************************************************/

from torch.nn import functional as F
import torch

# --- defines the loss function --- #
def get_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

    return BCE + KLD


