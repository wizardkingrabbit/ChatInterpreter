#!/usr/bin/python
# -*- coding: utf-8 -*-

# import all you need
from Utilities import * 
from Clip import * 
from Tokenizer_kit import * 
import pickle 
from Data_loader import * 
from Data_converter import * 
import torch
import os
from random import shuffle
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np 
from Embedding import * 
import gensim 

import warnings 
warnings.filterwarnings(action='ignore') 



if torch.cuda.is_available(): 
    device = torch.device('cuda:0') 
else: 
    device = torch.device('cpu') 
    



# Utilizes Pytorch's built in GRU function
# Kept training functions mostly the same as RNN, only changed the RNN class to GRU and updated based on that
# 
#
#


'''==================================================== Supportive methods ========================================''' 
gru_params={'n_epochs':400,
            'learning_rate':0.002, 
            'binary':True, 
            'test_ratio':0.2, 
            'hidden_size':200, 
            'num_layers':1
            }



# this function compute sigmoid gradient 
def Compute_sigmod_grad(v): 
    x = torch.tensor(v, requires_grad=True, dtype=torch.float) 
    y = 1/(1 + torch.exp(-x)) 
    y.backward()
    print(x.grad)
    return x.grad


# Split a list of clips into (train, test) tuple of lists with passed ratio
def Train_test_split(clip_list:list, ratio:float) -> tuple: 
    split_ind = int(len(clip_list)*ratio) 
    to_return = (list(), list()) 
    for i,clip in enumerate(clip_list): 
        if i>split_ind: 
            to_return[1].append(clip) 
        else: 
            to_return[0].append(clip) 
    return to_return


# GRU class 
class GRU(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, output_size:int):
        super(GRU, self).__init__() 
        self.hidden_size = hidden_size 
        self.input_size = input_size 
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)               # hidden_size bc of last linear layer before classification 
        self.soft_max = nn.LogSoftmax(dim=1) 

    def forward(self, input, hidden):
        #print(x.shape)
        output, hidden = self.gru(input, hidden)
        output = output[:, -1, :]           #reshape before passing to fully connected layer
        output = self.fc(output)
        output = self.soft_max(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)



# make a clip tuple to tensors tuple 
def Clip_to_tensor(clip:tuple) -> tuple: 
    chat_vec = clip[0] 
    label_int = clip[1]
    chat_tensor = torch.from_numpy(chat_vec) 
    label_tensor = torch.tensor([label_int,]) 
    return (chat_tensor, label_tensor) 

    
# Train on one clip 
def Train_gru_on_clip(gru:GRU, clip:tuple, lr=gru_params['learning_rate']): 
    clip = Clip_to_tensor(clip)
    criterion = nn.CrossEntropyLoss()
    ###optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
    ###optimizer.zero_grad()
    hidden = gru.initHidden()
    gru.zero_grad() 
    chat_tensor = clip[0]
    label_tensor = clip[1]
    #print("chat tensor size: ", chat_tensor.size())
    for i in range(chat_tensor.size()[0]): 
        x = chat_tensor[i][:,None,:]
        output, hidden = gru(x, hidden) 

    #print("output.size():", output.size())
    #print("label_tensor.size():",label_tensor.size(), ", label_tensor:", label_tensor)
    loss = criterion(output, label_tensor) 
    loss.backward()
    ###optimizer.step()
    for p in gru.parameters():
        p.data.add_(p.grad.data, alpha=-lr) 

    return output, loss.item() 


# Train on a list of clip data
def Train_gru(data_list:list, gru:GRU, n_epochs=gru_params['n_epochs'], lr=gru_params['learning_rate']) -> GRU: 
    epoch_ind = np.array([]) 
    for _ in range(n_epochs): 
        ind = np.arange(len(data_list)) 
        np.random.shuffle(ind)
        epoch_ind = np.concatenate((epoch_ind,ind)) 
        
    for i,ind in enumerate(epoch_ind): 
        clip = data_list[int(ind)] 
        output, loss = Train_gru_on_clip(gru, clip) 
        Print_progress(i,len(epoch_ind), message=f"Current loss: [{loss}]") 
        
    return gru 
        

# Predict the label of passed clip using passed gru 
def Predict(gru:GRU, clip:tuple) -> int: 
    clip = Clip_to_tensor(clip) 
    hidden = gru.initHidden()

	# Generate the input for gru 
    chat_tensor = clip[0]
    label_tensor = clip[1]
    for i in range(chat_tensor.size()[0]): 
        x = chat_tensor[i][:,None,:]
        output, hidden = gru(x, hidden) 
        
    topv, topi = output.topk(output.size()[1], dim=1, largest=True)
    softmax = nn.Softmax(dim=1)
    return int(topi[0][0]) 


# Calculate accuracy on a clip list, return tuple of (default accuracy, real accuracy)
def Prediction_accuracy(clip_list:list, gru:GRU, mislabeled=list()) -> float: 
    Y = np.array([i[1] for i in clip_list]) 
    Y_h = np.zeros(Y.shape) 
    n_default = float(np.sum(Y==Y_h))
    for i,clip in enumerate(clip_list): 
        Y_h[i] = Predict(gru,clip) 
        if Y_h[i]!=Y[i]: mislabeled.append(i)
    n_correct = float(np.sum(Y==Y_h))
    default_acc = n_default/len(clip_list) 
    test_acc = n_correct/len(clip_list)
    return (default_acc, test_acc)

'''==================================================== Main ================================================='''

def main(): 
    print(long_line)
    # clip_list = Prompt_for_data() 
    clip_list = Directory_load('./labeled_clip_data/Teo')
    if len(clip_list)==0:
        print("len(clip_list)==0")
    else: 
        print(f"Number of clips found: [{len(clip_list)}]")  
    print(f"Shuffling clips")
    shuffle(clip_list, Default_shuffle_func) 
    print(f"Splitting train and test on ratio: [{gru_params['test_ratio']}]")
    train_clips,test_clips = Train_test_split(clip_list, gru_params['test_ratio']) 
    
    mislabeled = list() 
    kv = Load_wv('word_vectors/teo_once.kv') 
    print(f"Processing train clips")
    train = Clip_list_2_rnn_data(train_clips, kv, gru_params['binary']) 
    print(f"processing test clips")
    test = Clip_list_2_rnn_data(test_clips, kv, gru_params['binary']) 
    # now train and test should be lists of tuple (chat 2d, label 1d) 

    gru = GRU(kv.vector_size, gru_params['hidden_size'], gru_params['num_layers'], (2 if gru_params['binary'] else 9)) 
    # print(f"Number of params is: [{len(gru.parameters())}]")
    print("Training...") 
    gru = Train_gru(train, gru).to(device)
    ind=list() 
    def_test_acc,test_acc = Prediction_accuracy(test, gru, ind) 
    mislabeled += [test_clips[i] for i in ind] 
    ind=list()
    def_train_acc,train_acc = Prediction_accuracy(train, gru, ind) 
    mislabeled += [train_clips[i] for i in ind]
    print(f"Default training accuracy is: [{def_train_acc}]")
    print(f"Training accuracy is: [{train_acc}]")
    print(f"Default test accuracy is: [{def_test_acc}]") 
    print(f"Test accuracy is: [{test_acc}]") 
    print(f"Total number of mislabeled clips is: [{len(mislabeled)}]") 
    
    file_path = prompt_for_save_file(dir_path='mislabeled', f_format='.pkl') 
    if file_path==None: 
        return
    with open(file_path, 'wb') as f:    
        pickle.dump(mislabeled, f) 
        print(f"Dumped [{len(mislabeled)}] clips into file")
    print(f"File saved as {file_path}") 
    return 
        
        
         
         
         
if __name__=='__main__': 
    main() 
    exit(0)