#!/usr/bin/python
# -*- coding: utf-8 -*-
# this is the template learner, do not change this file but make copies and name them accordingly


# import all you need
from __future__ import unicode_literals, print_function, division
import os 
from Utilities import * 
from Clip import * 
from Tokenizer_kit import * 
import pickle 
from Data_loader import * 
from io import open
import glob
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

import warnings 
warnings.filterwarnings(action='ignore') 

# ======================================= objects and methods ======================================= 


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
    return Normalize_vector(vector=vector) 


# vectorize a sentence
def Sentence_to_vector(sentence:str, kv:gensim.models.KeyedVectors) -> np.ndarray:  
    ''' Tokenize a sentence into a list of words to be vectorized''' 
    assert len(sentence)>0, "passed an empty sentence" 
    word_list = Embedding_tokenize(sentence=sentence) 
    if len(word_list)==0: 
        return np.zeros(kv.vector_size, dtype=float) 
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
def Clip_to_vector_sequential(clip:clip_it, 
                              kv:gensim.models.KeyedVectors, 
                              chat_window=1, 
                              overlap=0) -> np.ndarray:  
    
    ''' Turns a clip into a 2D array of vectors 
        chat window is how many chat to be considered a sentence''' 
    sentence_list = Clip_to_sentences(clip=clip, chat_window=chat_window, overlap=overlap) 
    #print(f"vectorizing sentence [{sentence_list[0]}]")
    to_return = Sentence_to_vector(sentence=sentence_list[0], kv=kv) 
    for s in sentence_list[1:]: 
        #print(f"vectorizing sentence [{s}]")
        to_return = np.vstack( (to_return,Sentence_to_vector(sentence=s, kv=kv)) ) 
    return to_return 
    

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		# Put the declaration of the RNN network here
		self.hidden_size = hidden_size 
		self.i2o = nn.Linear(input_size + hidden_size, output_size) 
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size) 
		self.soft_max = nn.LogSoftmax(dim=1) 

	def forward(self, input, hidden):
		# Put the computation for the forward pass here
		# combined is both the input and hidden, compute it so it do not have to be computed twice
		combined = torch.cat((input, hidden), dim=1)
		output = self.soft_max(self.i2o(combined)) 
		hidden = self.i2h(combined)

		return output, hidden

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


def Train_and_test_learner(learner, X:list, Y, test_ratio:float) -> tuple: # train the passed learner and return filed cases 
    ''' This function should train on passed X,Y data with test_ration split 
        X should be a list of clip objects, this is for the return value
        return is a two-tuple, first is result message with accuracies, second is list of mislabeled clips'''
    mislabeled_clips = list() 
    train_accuracy = 0.0 
    test_accuracy = 0.0 
    default_accuracy = 0.0 
    auc_score = 0.0 
    result_msg = (f"Total number of clips: [{len(X)}]" + os.linesep) 
    result_msg += (f"Train accuracy is: [{train_accuracy}]" + os.linesep) 
    result_msg += (f"Test accuracy is: [{test_accuracy}]" + os.linesep) 
    result_msg += (f"Default accuracy is: [{default_accuracy}]" + os.linesep) 
    result_msg += (f"AUC score is: [{auc_score}]" + os.linesep) 
    result_msg += (f"Number of mislabeled test clips: [{len(mislabeled_clips)}]")
    return (result_msg , mislabeled_clips)



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
            
        
        
        # change this part 
        msg,mislabeled = Train_and_test_learner(learner=None, X=clip_list, Y=None, test_ratio=0.2) 
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