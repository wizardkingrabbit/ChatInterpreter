#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np 
import sys 
import time 
import math 
from Utilities import Print_progress
import torch


if torch.cuda.is_available(): 
    device = torch.device('cuda:0') 
else: 
    device = torch.device('cpu') 
    
x=np.array([[0,0,0],[1,1,1]],dtype=np.float32) 
x=torch.from_numpy(x) 
x.to(device) 
print(x)
print(x.size())
print(x[0])
print(x[0].shape) 
print(x.dtype)
y=torch.tensor([2.0,]) 
print(y.size()) 
a = np.array([]) 
b = np.array([1,2,3]) 

print(np.concatenate((a,b)))
