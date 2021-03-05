import torch 
import pickle
import numpy as np 
import os 
import scipy 
from Data_converter import * 
from Data_loader import * 
from Utilities import * 
from Embedding import * 
from RNN import * 

'''
This module is not for utility use but just trials of clipping chronologically 
'''


vod_chrono = 'teo_chrono/teo6.pkl' 
kv_path = 'word_vectors/teo.kv' 
rnn = torch.load('rnn.pt')



with open(vod_chrono,'rb') as f: 
    data = pickle.load(f) 

interval = data[1][0]-data[0][0]
vod_intervals = np.zeros(len(data), dtype=np.float32) 
kv = Load_wv(kv_path)
print(f"processing data") 
chats = [i[1] for i in data] 
for i in range(len(chats)): 
    c = chats[i].split(os.linesep)[1:] 
    if len(c)<6: 
        vod_intervals[i]=0 
        Print_progress(i,len(chats),message='marked 0') 
        i+=1
        continue 
    c = [Token_list_to_vec(Embedding_tokenize(s), kv) for s in c] 
    c = np.array(c) 
    mean = np.mean(c,axis=0) 
    mean = mean/np.linalg.norm(mean) 
    c = c-mean 
    c = np.mean(np.linalg.norm(c,axis=1)) 
    if c>0.5: 
        vod_intervals[i]=0 
        Print_progress(i,len(chats),message='marked 0') 
    else: 
        vod_intervals[i]=1
        Print_progress(i,len(chats),message='marked 1') 
    i+=1

# chat_vec = np.expand_dims(chat_vec,axis=1) 
# chat_tensor = torch.from_numpy(chat_vec) 

# hidden = rnn.initHidden() 
# for i in range(chat_tensor.size()[0]): 
#         output, hidden = rnn(chat_tensor[i], hidden) 
        
#         topv, topi = output.topk(output.size()[1], dim=1, largest=True)
#         softmax = nn.Softmax(dim=1)
#         vod_intervals[i] = int(topi[0][0]) 
#         Print_progress(i,chat_tensor.size()[0])
        
# # for i in range(len(data)): 
# #     chat = data[i][1] 
# #     if chat=='': 
# #         vod_intervals[i]=0 
# #         continue
# #     chat_list = chat.split(os.linesep) 
# #     vec = Chat_to_2d_vec(chat_list=chat_list, kv=kv, window=1, overlap=0) 
# #     vec = np.vstack([vec for i in range(10)])
# #     # print(vec.shape)
# #     vec = np.expand_dims(vec,axis=1)
# #     vod_intervals[i] = Predict(rnn,(vec,0)) 
# #     Print_progress(i,len(data)) 
    
mask = np.ones(2*interval,dtype=np.float32) 
mask = mask/np.sum(mask) 
assert np.abs(np.sum(mask)-1)<0.1
print(f"prediction sum is [{np.sum(vod_intervals)}]/[{len(chats)}]")

vod_intervals = scipy.ndimage.correlate(vod_intervals,mask,mode='constant',cval=0) 
vod_intervals = (vod_intervals>0.1) 

clip_list = list() 
ind = 0 
start=0 
end=0 
clipping=False 
print(f"clipping vod")
while ind<vod_intervals.shape[0]: 
    if not(clipping or vod_intervals[ind]): 
        pass  
    elif not(clipping) and vod_intervals[ind]: 
        clipping=True 
        start=ind*interval 
    elif clipping and not(vod_intervals[ind]): 
        clipping=False 
        end=ind*interval 
        clip_list.append((int(start),int(end))) 
    
    Print_progress(ind,vod_intervals.shape[0])
    ind+=1 

results=list() 
first=True
start=0 
end=-1
clip_list = [(i[0]-20,i[1]+20) for i in clip_list]
for interval in clip_list:
    if interval[0]<=end: 
        end=results[-1][1]=interval[1] 
        continue 
    else: 
        start=interval[0] 
        end=interval[1] 
        clip=[int(start),int(end)]
        results.append(clip) 
        continue


print(f"clip for file [{vod_chrono}]")
print(f"number of clips found: [{len(results)}]") 
total_time=0
for i,interval in enumerate(results): 
    if interval[1]-interval[0]>180: 
        continue
    print(f"Clip number [{i+1}] is: [{time_to_str(interval[0])}] -> [{time_to_str(interval[1])}]") 
    total_time+=(interval[1]-interval[0])

    
print(f"total time to edit is [{time_to_str(total_time)}]")
     

