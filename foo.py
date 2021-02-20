import numpy as np 
import sys 
import time 
x=np.array([[0,1],
            [1,0],
            [0,1]]) 
y=np.array([[0,1],
            [0,1],
            [0,1]]) 
x=np.zeros(x.shape)
print(x)
z=np.array([0,1,0]) 
x[np.arange(x.shape[0]),z]=1.0 
print(x)
