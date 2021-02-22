#!/usr/bin/python
# -*- coding: utf-8 -*-
# this is the template learner, do not change this file but make copies and name them accordingly


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
import time
import math
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
    

'''==================================================== Supportive methods ========================================''' 
rnn_params={'n_epochs':400,
            'learning_rate':0.002, 
            'binary':True, 
            'test_ratio':0.2, 
            'hidden_size':90, 
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


# RNN class 
class RNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(RNN, self).__init__() 
        self.hidden_size = hidden_size 
        self.input_size = input_size 
        self.i2o = nn.Linear(input_size + hidden_size, output_size) 
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) 
        self.soft_max = nn.LogSoftmax(dim=1) 


    def forward(self, input, hidden):
        # Put the computation for the forward pass here
        # combined is both the input and hidden, compute it so it do not have to be computed twice 
        combined = torch.cat((input, hidden),dim=1)
        output = self.soft_max(self.i2o(combined)) 
        hidden = self.i2h(combined)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)



# make a clip tuple to tensors tuple 
def Clip_to_tensor(clip:tuple) -> tuple: 
    chat_vec = clip[0] 
    label_int = clip[1]
    chat_tensor = torch.from_numpy(chat_vec) 
    label_tensor = torch.tensor([label_int,]) 
    return (chat_tensor, label_tensor) 

    
# Train on one clip 
def Train_rnn_on_clip(rnn:RNN, clip:tuple, lr=rnn_params['learning_rate']): 
    clip = Clip_to_tensor(clip)
    criterion = nn.CrossEntropyLoss()
    hidden = rnn.initHidden()
    rnn.zero_grad() 
    chat_tensor = clip[0]
    label_tensor = clip[1]
    for i in range(chat_tensor.size()[0]): 
        output, hidden = rnn(chat_tensor[i], hidden) 

    loss = criterion(output, label_tensor) 
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-lr) 

    return output, loss.item() 


# Train on a list of clip data
def Train_rnn(data_list:list, rnn:RNN, n_epochs=rnn_params['n_epochs'], lr=rnn_params['learning_rate']) -> RNN: 
    epoch_ind = np.array([]) 
    for _ in range(n_epochs): 
        ind = np.arange(len(data_list)) 
        np.random.shuffle(ind)
        epoch_ind = np.concatenate((epoch_ind,ind)) 
        
    for i,ind in enumerate(epoch_ind): 
        clip = data_list[int(ind)] 
        output, loss = Train_rnn_on_clip(rnn, clip) 
        Print_progress(i/len(epoch_ind), message=f"Current loss: [{loss}]") 
        
    return rnn 
        

# Predict the label of passed clip using passed rnn 
def Predict(rnn:RNN, clip:tuple) -> int: 
    clip = Clip_to_tensor(clip) 
    hidden = rnn.initHidden()

	# Generate the input for RNN 
    chat_tensor = clip[0]
    label_tensor = clip[1]
    for i in range(chat_tensor.size()[0]): 
        output, hidden = rnn(chat_tensor[i], hidden) 
        
    topv, topi = output.topk(output.size()[1], dim=1, largest=True)
    softmax = nn.Softmax(dim=1)
    return int(topi[0][0]) 


# Calculate accuracy on a clip list, return tuple of (default accuracy, real accuracy)
def Prediction_accuracy(clip_list:list, rnn:RNN, mislabeled=list()) -> float: 
    Y = np.array([i[1] for i in clip_list]) 
    Y_h = np.zeros(Y.shape) 
    n_default = float(np.sum(Y==Y_h))
    for i,clip in enumerate(clip_list): 
        Y_h[i] = Predict(rnn,clip) 
        if Y_h[i]!=Y[i]: mislabeled.append(clip_list[i])
    n_correct = float(np.sum(Y==Y_h))
    default_acc = n_default/len(clip_list) 
    test_acc = n_correct/len(clip_list)
    return (default_acc, test_acc)

'''==================================================== Main ================================================='''

def main(): 
    while(True): # keeps prompting for and train on data 
        print(long_line)
        # clip_list = Prompt_for_data() 
        clip_list = Directory_load('./labeled_clip_data/Teo')
        if len(clip_list)==0:
            break 
        else: 
            print(f"Number of clips found: [{len(clip_list)}]")  
        print(f"Shuffling clips")
        shuffle(clip_list, Default_shuffle_func) 
        print(f"Splitting train and test on ratio: [{rnn_params['test_ratio']}]")
        train,test = Train_test_split(clip_list, rnn_params['test_ratio']) 
        
        mislabeled = list() 
        kv = Load_wv('word_vectors/teo.kv') 
        print(f"Processing clips")
        train = Clip_list_2_rnn_data(train, kv, rnn_params['binary']) 
        test = Clip_list_2_rnn_data(test, kv, rnn_params['binary']) 
        # now train and test should be lists of tuple (chat 2d, label 1d) 
        rnn = RNN(kv.vector_size, rnn_params['hidden_size'], (2 if rnn_params['binary'] else 9)) 
        # for i in rnn.parameters(): print(i.size())
        # print(f"Number of params is: [{len(rnn.parameters())}]")
        print("Training...") 
        rnn = Train_rnn(train, rnn) 
        def_test_acc,test_acc = Prediction_accuracy(test, rnn, mislabeled) 
        def_train_acc,train_acc = Prediction_accuracy(train, rnn, mislabeled) 
        print(f"Default training accuracy is: [{def_train_acc}]")
        print(f"Training accuracy is: [{train_acc}]")
        print(f"Default test accuracy is: [{def_test_acc}]") 
        print(f"Test accuracy is: [{test_acc}]") 
        print(f"Total number of mislabeled clips is: [{len(mislabeled)}]") 
        
        file_path = prompt_for_save_file(dir_path='mislabeled', f_format='.pkl') 
        if file_path==None: 
            continue
        with open(file_path, 'wb') as f: 
            pickle.dump(mislabeled, f) 
        print(f"File saved as {file_path}") 
        continue
        
        
    return 
        
        
         
         
         
if __name__=='__main__': 
    main() 
    exit(0)