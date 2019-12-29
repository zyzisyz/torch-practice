#!/bin/bash

#*************************************************************************
#	> File Name: train.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Sun Dec 29 10:58:36 2019
# ************************************************************************/

python -u train.py \
	--batch-size 100 \
	--epochs 10 \
	--test-batch-size 1000 \
	--lr 0.01 \
	--momentum 0.5 \
	--save-model 
