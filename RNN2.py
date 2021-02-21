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

rnn_params={'kv':None, 
            'chat_window':1, 
            'overlap':0,
            'n_epochs':5,
            'learning_rate':0.005, 
            'binary':True, 
            'test_ratio':0.25, 
            'hidden_size':128, 
            'input_size':300,
            'batch_size':15,
            'num_layers':1,
            'sequence_length':10
            }

# #hyperparameters
# input_size = 300
# hidden_size = 128
# num_classes = 2
# num_epochs = 5
# batch_size = 15
# learning_rate = .005
# num_layers = 1
# sequence_length = 10

# this function compute sigmoid gradient 
def Compute_sigmod_grad(v): 
    x = torch.tensor(v, requires_grad=True, dtype=torch.float) 
    y = 1/(1 + torch.exp(-x)) 
    y.backward()
    print(x.grad)
    return x.grad


# turn a word into numpy vector
def Word_to_vector(word:str, kv:gensim.models.KeyedVectors) -> np.ndarray: 
    ''' Turns a word into tensor vector, returns 0 if word not in vocab'''
    vector = Vector_of(word_vector=kv, word=word) 
    if type(vector)!=np.ndarray: 
        return np.zeros(kv.vector_size, dtype=float) 
    return vector
    
    
# compute magnitude of vector 
def Mag_of(vector:np.ndarray): 
    return np.sqrt(vector.dot(vector)) 


# normalize a vector 
def Normalize_vector(vector:np.ndarray) -> np.ndarray: 
    ''' This function normalize the passed vector to have magnitude of 1''' 
    return vector/(Mag_of(vector)+np.finfo(dtype=float).eps)
    

# vectorize a list of words
def List_to_vector(word_list:list, kv:gensim.models.KeyedVectors) -> np.ndarray:  
    ''' Turns a list of word into a numpy vector, words not in vocab are counted as 0 
        normalize the resulting vector and return'''
    vector = 0.0 
    assert len(word_list)>0, "Cannot vectorize empty list"
    for word in word_list: 
        vector = vector + Word_to_vector(word, kv) 
        vector = np.array(vector, dtype=np.double)
    return Normalize_vector(vector=vector) 


# vectorize a sentence
def Sentence_to_vector(sentence:str, kv:gensim.models.KeyedVectors) -> np.ndarray:  
    ''' Tokenize a sentence into a list of words to be vectorized''' 
    assert len(sentence)>0, "passed an empty sentence" 
    word_list = Embedding_tokenize(sentence=sentence) 
    if len(word_list)==0: 
        return np.zeros(kv.vector_size, dtype=np.double) 
    vector = List_to_vector(word_list=word_list, kv=kv) 
    return vector 
    
    
# turns the chats of one clip into list of sentences 
def Clip_to_sentences(clip:clip_it, chat_window=1, overlap=0) -> list: 
    ''' turns a clip of chats into list of sentences, 
        chat window is how many chats we consider a sentence 
        overlap is the overlap between chat windows when we shift ''' 
    assert overlap<chat_window, "overlap too large" 
    sentence_list = list() 
    chat_list = clip.chats 
    window_index = 0 
    shift_distance = chat_window - overlap 
    while(window_index<len(chat_list)): 
        sentence = Concatenate_str_list(str_list=chat_list[window_index:window_index+chat_window]) 
        window_index += shift_distance 
        sentence_list.append(sentence) 
    return sentence_list 
        

# Turn a clip object into numpy vector
def Clip_to_vector_sequential(clip_converter:dict, clip:clip_it) -> np.ndarray:  
    ''' Turns a clip into a 2D array of vectors 
        chat window is how many chat to be considered a sentence''' 
    kv = clip_converter['kv'] 
    assert type(kv)==gensim.models.KeyedVectors, "passed kv not valid"
    chat_window = clip_converter['chat_window'] 
    overlap = clip_converter['overlap'] 
    sentence_list = Clip_to_sentences(clip=clip, chat_window=chat_window, overlap=overlap) 
    #print(f"vectorizing sentence [{sentence_list[0]}]")
    to_return = Sentence_to_vector(sentence=sentence_list[0], kv=kv) 
    if len(sentence_list)==1: 
        to_return = np.transpose(to_return)
    for s in sentence_list[1:]: 
        #print(f"vectorizing sentence [{s}]")
        to_return = np.vstack( (to_return,Sentence_to_vector(sentence=s, kv=kv)) ) 
    return to_return 
    

# turn clip label into a vector
def Clip_to_category_vector(clip:clip_it, binary=rnn_params['binary']) -> np.ndarray: 
    if binary: 
        label = clip.get_label_binary() 
        if label!=0: 
            return np.array([0,1], dtype=float) 
        else: 
            return np.array([1,0], dtype=float) 
    else: 
        label = clip.get_label() 
        to_return = np.zeros(len(clip.available_labels), dtype=float) 
        to_return[label]=1.0 
        return to_return


# turn a category vector into label int 
def Category_tensor_to_label(vector:torch.Tensor) -> int: 
    ''' This function takes a tensor vector and convert it to a label integer''' 
    for i in range(len(vector)): 
        if vector[i].item()==1: 
            return i 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        # Put the declaration of the RNN network here
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)            # input shape needs to be (batch size, sequence_length, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)            # output shape (batch_siuze, sequence_length, input_size)
        out = out[:, -1, :]
        out = self.fc(out)

        return out



# ====================================== end of objects and methods ====================================
def Train_iter_rnn(rnn:RNN, category_tensor:torch.Tensor, sentences_tensor:torch.Tensor, learning_rate=rnn_params['learning_rate']):   
    ''' train the passed rnn module on a category tensor and sentences tensor''' 
    assert sentences_tensor.size()[1] == rnn.input_size, "Different size of input"
    criterion = nn.NLLLoss()
    hidden = rnn.initHidden()
    rnn.zero_grad() 
    
    for i in range(sentences_tensor.size()[0]):
        output, hidden = rnn(sentences_tensor[i], hidden) 

    loss = criterion(output, category_tensor) 
    loss.backward() 
    
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate) 

    return output, loss.item() 


# train passed rnn with passed clip 
def Train_rnn_on_clip(rnn:RNN, 
                clip:clip_it, 
                clip_converter=rnn_params,
                learning_rate=rnn_params['learning_rate'], 
                binary=rnn_params['binary']): 
    ''' train passed rnn on a clip, do clip processing itself with passed params'''
    sentences_vector = Clip_to_vector_sequential(clip_converter=clip_converter, clip=clip) 
    label_vector = Clip_to_category_vector(clip=clip, binary=binary) 
    sentences_tensor = torch.from_numpy(sentences_vector).double()
    category_tensor = torch.from_numpy(label_vector).double()
    return Train_iter_rnn(rnn=rnn, category_tensor=category_tensor, sentences_tensor=sentences_tensor, learning_rate=learning_rate) 
    

# returns the time since passed value in formated string
def Time_since(since): 
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60 
    return f"[{m}m:{s}s]"


# print a message every specified number of times
def Print_every(n:int, every:int, message:str): 
    if n % every == 0: 
        print(f"Trained number of clips: [{n}]") 
        print(message) 
    return 


# train an rnn using passed clip list 
def Train_rnn(clip_list:list, rnn:RNN, params=rnn_params) -> RNN: 
    ''' train passed rnn on passed clip list using specified params
        return trained rnn object''' 
    n_epochs = params['n_epochs'] 
    learning_rate=params['learning_rate'] 
    binary=params['binary']
    print_every=50 
    current_loss=0.0
    n_trained=0
    start = time.time() 
    
    for epoch in range(n_epochs): 
        print(short_line)
        print(f"Epoch [{epoch}/{n_epochs}]: ")
        order_ind = np.arange(len(clip_list)) 
        np.random.shuffle(order_ind) 
        for i in order_ind: 
            output,loss = Train_rnn_on_clip(rnn=rnn, clip=clip_list[i], clip_converter=params, learning_rate=learning_rate, binary=binary) 
            current_loss+=loss 
            n_trained+=1
            message = f"Time passed: {Time_since(start)}"+os.linesep
            message += f"Current loss is [{current_loss}]" 
            Print_every(n_trained, print_every, message) 
            
    return rnn 
        

# predict a clip using passed rnn 
def Predict(rnn:RNN, clip:clip_it, params=rnn_params) -> int: 
    hidden = rnn.initHidden() 
    sentences_tensor = torch.from_numpy(Clip_to_vector_sequential(params, clip)) 
    category_tensor = torch.from_numpy(Clip_to_category_vector(clip, binary=params['binary'])) 
    
    for i in range(sentences_tensor.size()[0]):
        output, hidden = rnn(sentences_tensor[i], hidden) 
    prediction = torch.argmax(output).item() 
    return prediction 
    

# test rnn on a single clip, return boolean
def Test_rnn_on_clip(rnn:RNN, clip:clip_it, params=rnn_params) -> bool: 
    prediction = Predict(rnn, clip, params) 
    if params['binary']: 
        clip.set_pred_binary_label(prediction) 
        return prediction==(clip.get_label_binary()**2) 
    else: 
        clip.set_pred_label(prediction) 
        return prediction==clip.get_label() 
        

# test trained learner on a list of clips, returns mislabeled clips
def Test_rnn(rnn:RNN, clip_list:list, params=rnn_params) -> list:
    to_return = list() 
    for clip in clip_list: 
        if not(Test_rnn_on_clip(rnn, clip, params)): 
            to_return.append(clip) 
    return to_return 




# train the passed learner and return filed cases 
def Train_and_test_learner(learner, X:list, Y=None, params=rnn_params) -> tuple: 
    ''' This function should train on passed X,Y data with test_ration split 
        X should be a list of clip objects, this is for the return value
        return is a two-tuple, first is result message with accuracies, second is list of mislabeled clips''' 
    test_ratio=params['test_ratio']
    mislabeled_clips = list() 
    random_ind = np.arange(len(X)) 
    np.random.shuffle(random_ind) 
    X_train = list() 
    X_test = list() 
    cut_ind = int(len(X)*test_ratio) 
    for i in range(len(X)): 
        if i<cut_ind: 
            X_test.append(X[i].copy()) 
        else: 
            X_train.append(X[i].copy()) 
    
    learner = Train_rnn(clip_list=X_train, rnn=learner, params=params) 
    
    mis_test_clips = Test_rnn(learner, X_test, params)
    mis_training_clips = Test_rnn(learner, X_train, params) 
    train_accuracy = len(mis_training_clips)/float(len(X_train)) 
    test_accuracy = len(mis_test_clips)/float(len(X_test)) 
    mislabeled_clips = mis_test_clips + mis_training_clips 
    n_default=0
    for clip in X: 
        if params['binary'] and (clip.get_label_binary()==0): 
            n_default+=1 
        elif (clip.get_label()==1): 
            n_default+=1 
            
    default_accuracy = float(n_default)/len(X) 
    result_msg = (f"Total number of clips: [{len(X)}]" + os.linesep) 
    result_msg += (f"Train accuracy is: [{train_accuracy}]" + os.linesep) 
    result_msg += (f"Test accuracy is: [{test_accuracy}]" + os.linesep) 
    result_msg += (f"Default accuracy is: [{default_accuracy}]" + os.linesep) 
    result_msg += (f"Number of mislabeled test clips: [{len(mislabeled_clips)}]")
    return (result_msg, mislabeled_clips) 


# This function prompt for params of rnn 
def Prompt_for_rnn_params(): 
    rnn_params['chat_window'] = prompt_for_int("Enter chat window in int: ", min_v=1) 
    rnn_params['overlap'] = prompt_for_int("Enter overlap in int: ", min_v=0) 
    rnn_params['n_epoch'] = prompt_for_int("Enter number of epoch: ", min_v=1) 
    ans = prompt_for_str("Train on binary? (y/n): ", options={'y','n'}) 
    if ans=='n': 
        rnn_params['binary']=False 
    rnn_params['learning_rate'] = prompt_for_float("Enter learning rate in float: ", min_v=0.005)
    rnn_params['test_ratio'] = prompt_for_float("Enter the test ratio: ", min_v=0.1)
    rnn_params['hidden_size'] = prompt_for_int("Enter hidden size: ", min_v=10) 

    
# ====================================== end of objects and methods ====================================

def main(): 
    while(True): # keeps prompting for and train on data 
        print(long_line)
        clip_list = Prompt_for_data() 
        if len(clip_list)==0:
            break 
        # delete this part if you want
        else: 
            print(f"Number of clips found: [{len(clip_list)}]")  
            
        rnn_params['kv'] = Load_wv() 
        ans = prompt_for_str("Do you want to use default params? (y/n): ", options={'y','n'})
        if ans=='n': 
            Prompt_for_rnn_params() 
        
        if rnn_params['binary']: 
            output_size=2 
        else: 
            output_size=9
        learner = RNN(input_size=rnn_params['kv'].vector_size, hidden_size=rnn_params['hidden_size'], output_size=output_size)
        
        # change this part 
        msg,mislabeled = Train_and_test_learner(learner=learner, X=clip_list) 
        print(f"Printing out test result: ") 
        print(short_line)
        print(msg) 
        
        file_path = prompt_for_save_file(dir_path='mislabeled', f_format='.pkl') 
        if file_path==None: 
            continue
        with open(file_path, 'wb') as f: 
            pickle.dump(mislabeled, f) 
        print(f"File saved as {file_path}") 
        continue
            
    return 


# ====================================== user prompts =============================================== 


if __name__ == '__main__': 
    main() 
    exit(0) 