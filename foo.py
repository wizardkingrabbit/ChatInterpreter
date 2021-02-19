import numpy as np 
import torch 

x = np.array([1,2,3,4]) 
# print(x) 
# y = 0.0 + x 
# print(y) 
# z = x + x 
# print(z) 
# a = np.vstack([x,x,x]) 
# print(a)
# print(x.shape) 
# print(a[0])
# # (n,) [i,i,i,i] 
# # (1,n) [[i,i,i,i], ] 
# print(np.vstack((z,x))) 
y = torch.from_numpy(x) 
y=torch.transpose(y) 
print(len(y.size())) 
print(y)
# print(y.get_shape().as_list())
# for i in range(4): 
#     print(x[i])
#     print(y[i])
#     print(y[i].item())