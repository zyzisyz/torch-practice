#!/bin/bash

#*************************************************************************
#	> File Name: train.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Sun Dec 29 12:42:51 2019
# ************************************************************************/


python -u train.py \
	--batch-size 100 \
	--epochs 10 \
	--z_dim 30 \
	--lr 1e-3

