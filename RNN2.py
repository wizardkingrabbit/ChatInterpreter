#!/usr/bin/python
# -*- coding: utf-8 -*-
# this is the template learner, do not change this file but make copies and name them accordingly


# import all you need
from Utilities import * 
from Clip import * 
from Tokenizer_kit import * 
import pickle 
from Data_loader import * 
import torch
import os
import random
import time
import math
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np 
from Embedding import * 
import gensim 
from Data_converter import *
from sklearn.model_selection import train_test_split 

import warnings 
warnings.filterwarnings(action='ignore') 

device = torch.device('cpu')

# ======================================= objects and methods ======================================= 

# #hyperparameters
num_classes = 2             # defaulting to binary 
num_epochs = 5
batch_size = 10
learning_rate = 0.005

input_size = 300
sequence_length = 10
hidden_size = 128
num_layers = 1
binary = True

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        # Put the declaration of the RNN network here
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)            # input shape needs to be (batch size, sequence_length, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        print(x.size(0))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)           # x.size(0) is batch size

        out, _ = self.rnn(x, h0)            # output shape (batch_siuze, sequence_length, input_size)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

clip_list = Prompt_for_data() 
print(f"Number of clips found: [{len(clip_list)}]")  
    
kv = Load_wv()
data = Clip_list_2_rnn_data(clip_list, kv, binary)          
learner = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# turn data into correct batch size
inputs = list()
temp_inputs = np.zeros(batch_size, data[0].shape[0], i[0].shape[1])

for i in range(len(temp_inputs)):
    inputs[i] = data[i]



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)  


n_total_steps = len(data)
for epoch in range(num_epochs):
    for i in data:  
        # need to be shape (batch size, shape[0], shape[1])
        #inputs = clips.reshape(-1, sequence_length, input_size).to(device)
        inputs = np.zeros(batch_size, i[0].shape[0], i[0].shape[1])
        inputs = i[0]
        inputs = np.expand_dims(inputs, axis=0)
        print(inputs.shape)
        inputs = torch.from_numpy(inputs).to(device)
        labels = torch.from_numpy(i[1]).to(device)
        
        # Forward pass
        outputs = learner(inputs)
        loss = criterion(outputs, labels[0])
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')



# rnn_params={'kv':None, 
#             'chat_window':1, 
#             'overlap':0,
#             'n_epochs':5,
#             'learning_rate':0.005, 
#             'binary':True, 
#             'test_ratio':0.25, 
#             'hidden_size':128, 
#             }


# class RNN(nn.Module):
#     def __init__(self, input_size:int, hidden_size:int, output_size:int):
#         super(RNN, self).__init__() 
#         self.hidden_size = hidden_size 
#         self.input_size = input_size 
#         self.i2o = nn.Linear(input_size + hidden_size, output_size) 
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size) 
#         self.soft_max = nn.LogSoftmax(dim=1) 


#     def forward(self, input, hidden):
#         # Put the computation for the forward pass here
#         # combined is both the input and hidden, compute it so it do not have to be computed twice 
#         combined = torch.cat((input.double(), hidden.double()))
		
#         output = self.soft_max(self.i2o(combined)) 
#         hidden = self.i2h(combined)

#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(self.hidden_size)


# def Train_iter_rnn(rnn:RNN, category_tensor:torch.Tensor, sentences_tensor:torch.Tensor, learning_rate=rnn_params['learning_rate']):   
#     ''' train the passed rnn module on a category tensor and sentences tensor''' 
#     assert sentences_tensor.size()[1] == rnn.input_size, "Different size of input"
#     criterion = nn.NLLLoss()
#     hidden = rnn.initHidden()
#     rnn.zero_grad() 
    
#     for i in range(sentences_tensor.size()[0]):
#         output, hidden = rnn(sentences_tensor[i], hidden) 

#     loss = criterion(output, category_tensor) 
#     loss.backward() 
    
#     for p in rnn.parameters():
#         p.data.add_(p.grad.data, alpha=-learning_rate) 

#     return output, loss.item() 


# # train passed rnn with passed clip 
# def Train_rnn_on_clip(rnn:RNN, 
#                 clip:clip_it, 
#                 clip_converter=rnn_params,
#                 learning_rate=rnn_params['learning_rate'], 
#                 binary=rnn_params['binary']): 
#     ''' train passed rnn on a clip, do clip processing itself with passed params'''
#     sentences_vector = Clip_to_vector_sequential(clip_converter=clip_converter, clip=clip) 
#     label_vector = Clip_to_category_vector(clip=clip, binary=binary) 
#     sentences_tensor = torch.from_numpy(sentences_vector).double()
#     category_tensor = torch.from_numpy(label_vector).double()
#     return Train_iter_rnn(rnn=rnn, category_tensor=category_tensor, sentences_tensor=sentences_tensor, learning_rate=learning_rate) 
    

# # returns the time since passed value in formated string
# def Time_since(since): 
#     now = time.time()
#     s = now - since
#     m = math.floor(s / 60)
#     s -= m * 60 
#     return f"[{m}m:{s}s]"


# # print a message every specified number of times
# def Print_every(n:int, every:int, message:str): 
#     if n % every == 0: 
#         print(f"Trained number of clips: [{n}]") 
#         print(message) 
#     return 


# # train an rnn using passed clip list 
# def Train_rnn(clip_list:list, rnn:RNN, params=rnn_params) -> RNN: 
#     ''' train passed rnn on passed clip list using specified params
#         return trained rnn object''' 
#     n_epochs = params['n_epochs'] 
#     learning_rate=params['learning_rate'] 
#     binary=params['binary']
#     print_every=50 
#     current_loss=0.0
#     n_trained=0
#     start = time.time() 
    
#     for epoch in range(n_epochs): 
#         print(short_line)
#         print(f"Epoch [{epoch}/{n_epochs}]: ")
#         order_ind = np.arange(len(clip_list)) 
#         np.random.shuffle(order_ind) 
#         for i in order_ind: 
#             output,loss = Train_rnn_on_clip(rnn=rnn, clip=clip_list[i], clip_converter=params, learning_rate=learning_rate, binary=binary) 
#             current_loss+=loss 
#             n_trained+=1
#             message = f"Time passed: {Time_since(start)}"+os.linesep
#             message += f"Current loss is [{current_loss}]" 
#             Print_every(n_trained, print_every, message) 
            
#     return rnn 
        

# # predict a clip using passed rnn 
# def Predict(rnn:RNN, clip:clip_it, params=rnn_params) -> int: 
#     hidden = rnn.initHidden() 
#     sentences_tensor = torch.from_numpy(Clip_to_vector_sequential(params, clip)) 
#     category_tensor = torch.from_numpy(Clip_to_category_vector(clip, binary=params['binary'])) 
    
#     for i in range(sentences_tensor.size()[0]):
#         output, hidden = rnn(sentences_tensor[i], hidden) 
#     prediction = torch.argmax(output).item() 
#     return prediction 
    

# # test rnn on a single clip, return boolean
# def Test_rnn_on_clip(rnn:RNN, clip:clip_it, params=rnn_params) -> bool: 
#     prediction = Predict(rnn, clip, params) 
#     if params['binary']: 
#         clip.set_pred_binary_label(prediction) 
#         return prediction==(clip.get_label_binary()**2) 
#     else: 
#         clip.set_pred_label(prediction) 
#         return prediction==clip.get_label() 
        

# # test trained learner on a list of clips, returns mislabeled clips
# def Test_rnn(rnn:RNN, clip_list:list, params=rnn_params) -> list:
#     to_return = list() 
#     for clip in clip_list: 
#         if not(Test_rnn_on_clip(rnn, clip, params)): 
#             to_return.append(clip) 
#     return to_return 




# # train the passed learner and return filed cases 
# def Train_and_test_learner(learner, X:list, Y=None, params=rnn_params) -> tuple: 
#     ''' This function should train on passed X,Y data with test_ration split 
#         X should be a list of clip objects, this is for the return value
#         return is a two-tuple, first is result message with accuracies, second is list of mislabeled clips''' 
#     test_ratio=params['test_ratio']
#     mislabeled_clips = list() 
#     random_ind = np.arange(len(X)) 
#     np.random.shuffle(random_ind) 
#     X_train = list() 
#     X_test = list() 
#     cut_ind = int(len(X)*test_ratio) 
#     for i in range(len(X)): 
#         if i<cut_ind: 
#             X_test.append(X[i].copy()) 
#         else: 
#             X_train.append(X[i].copy()) 
    
#     learner = Train_rnn(clip_list=X_train, rnn=learner, params=params) 
    
#     mis_test_clips = Test_rnn(learner, X_test, params)
#     mis_training_clips = Test_rnn(learner, X_train, params) 
#     train_accuracy = len(mis_training_clips)/float(len(X_train)) 
#     test_accuracy = len(mis_test_clips)/float(len(X_test)) 
#     mislabeled_clips = mis_test_clips + mis_training_clips 
#     n_default=0
#     for clip in X: 
#         if params['binary'] and (clip.get_label_binary()==0): 
#             n_default+=1 
#         elif (clip.get_label()==1): 
#             n_default+=1 
            
#     default_accuracy = float(n_default)/len(X) 
#     result_msg = (f"Total number of clips: [{len(X)}]" + os.linesep) 
#     result_msg += (f"Train accuracy is: [{train_accuracy}]" + os.linesep) 
#     result_msg += (f"Test accuracy is: [{test_accuracy}]" + os.linesep) 
#     result_msg += (f"Default accuracy is: [{default_accuracy}]" + os.linesep) 
#     result_msg += (f"Number of mislabeled test clips: [{len(mislabeled_clips)}]")
#     return (result_msg, mislabeled_clips) 


# # This function prompt for params of rnn 
# def Prompt_for_rnn_params(): 
#     rnn_params['chat_window'] = prompt_for_int("Enter chat window in int: ", min_v=1) 
#     rnn_params['overlap'] = prompt_for_int("Enter overlap in int: ", min_v=0) 
#     rnn_params['n_epoch'] = prompt_for_int("Enter number of epoch: ", min_v=1) 
#     ans = prompt_for_str("Train on binary? (y/n): ", options={'y','n'}) 
#     if ans=='n': 
#         rnn_params['binary']=False 
#     rnn_params['learning_rate'] = prompt_for_float("Enter learning rate in float: ", min_v=0.005)
#     rnn_params['test_ratio'] = prompt_for_float("Enter the test ratio: ", min_v=0.1)
#     rnn_params['hidden_size'] = prompt_for_int("Enter hidden size: ", min_v=10) 

    
# # ====================================== end of objects and methods ====================================

# def main(): 
#     clip_list = Prompt_for_data() 
#     print(f"Number of clips found: [{len(clip_list)}]")  
        
#     kv = Load_wv()
#     data = Clip_list_2_rnn_data(clip_list, kv)              # default binary=True


#     learner = RNN(input_size=rnn_params['kv'].vector_size, hidden_size=rnn_params['hidden_size'], output_size=output_size).to(device)
    
#     # change this part 
#     msg,mislabeled = Train_and_test_learner(learner=learner, X=clip_list) 
#     print(f"Printing out test result: ") 
#     print(short_line)
#     print(msg) 
    
#     file_path = prompt_for_save_file(dir_path='mislabeled', f_format='.pkl') 
#     if file_path==None: 
#         continue
#     with open(file_path, 'wb') as f: 
#         pickle.dump(mislabeled, f) 
#     print(f"File saved as {file_path}") 
#     continue
            
#     return 

    
# ====================================== end of objects and methods ====================================



# ====================================== user prompts =============================================== 


if __name__ == '__main__': 
    main() 
    exit(0) 