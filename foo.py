#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle 
import numpy as np 
import scipy 
import json
from Clipper import *
from Data_loader import * 
from Data_converter import * 


# x=np.array([[1,2,3],
#             [2,2,2], 
#             [3,2,1],
#             [1,1,1]])

# print(scipy.stats.entropy(x,axis=1)) 


# x=np.array([1,2,3]) 
# y=np.array([[1,2,3], 
#             [3,3,3], 
#             [4,4,4]]) 
# print(np.linalg.norm(y,axis=1)) 
# print(np.linalg.norm(x)) 

x=y=a=1 
print(x,y,a)
# files = [os.path.join('chatjsonfiles/',i) for i in os.listdir('chatjsonfiles/') if 'Teo' in i] 
# print(files) 

# dest = 'teo_chrono'
# for i,f in enumerate(files): 
#     data = Load_json(f) 
#     data = Organize_chats_chrono(data) 
#     print(f"processed file {f}") 
#     print(f"intervals found: [{len(data)}]")
#     with open(os.path.join(dest,f"teo{i}.pkl"),'wb') as f: 
#         pickle.dump(data,f) 