#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: 01_np_torch.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Tue 07 Jan 2020 02:59:29 PM CST
# ************************************************************************/


import torch 
import numpy as np

# (2, 3)
np_data = np.arange(6).reshape((2, 3))

torch_data = torch.from_numpy(np_data)

tensor2np = torch_data.numpy()

print(np_data, "\n")
print(torch_data, "\n")
print(tensor2np)

# abs 
data = [-1, -2 ,1 , 2]
print(data)
tensor = torch.abs(torch.FloatTensor(data))
print(tensor.reshape(2, 2))

# matmul
data = [-1, -2 ,1 , 2]
data = np.array(data)
tensor = torch.matmul(torch.from_numpy(data), torch.from_numpy(data))

# output det(matrix)?
print(tensor)


