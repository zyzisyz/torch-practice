#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: loss.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sun Dec 29 12:20:37 2019
# ************************************************************************/


import torch.nn.functional as F

def get_loss(output, target):
	loss = F.nll_loss(output, target)
	return loss

