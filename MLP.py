#!/usr/bin/python
# -*- coding: utf-8 -*-
# this is the template learner, do not change this file but make copies and name them accordingly


# import all you need
import os 
from Utilities import * 
from Clip import * 
from Tokenizer_kit import * 
import pickle 
from Data_loader import *
from Data_converter import *
import random 
from Embedding import * 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import warnings 
warnings.filterwarnings(action='ignore') 

''' 
============================== MLP description ======================================
This module tests the accracy of Multi Layered Perceptron on prompted data. 
'''


'''======================================= objects and methods =======================================''' 

mlp_prms = {'binary':True, 
            'hidden':(100,), 
            'test_ratio':0.15, 
            'chat_threshold':0.75}

# Because of technical difficulties, you do not have use this function
# train the passed learner and return failed cases and print result message
def Train_and_test_learner(learner:MLPClassifier, clip_list:list, test_ratio:float) -> tuple:  
    ''' This function takes a learner, a clip list, and a test ratio 
        split the data set into trin and test, output the result and list of mislabeled clips from all clips''' 
    mislabeled_clips=list() 
    X,Y = Clip_list_2_mlp_data(clip_list, Load_wv()) 
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=mlp_prms['test_ratio'],shuffle=False) 
    Y_def = np.zeros(Y.shape) 
    Y_def[:,0]=1 
    print(f"Deault accuracy is: [{np.sum((Y==Y_def).all(axis=1))/Y.shape[0]}]") 
    print(f"Working with test ratio: [{test_ratio}]")
    print(f"In training, totally [{Y_train.shape[0]}] clips: ") 
    print(f">> number of class 1: [{np.sum((Y_train==[0,1]).all(axis=1))}]")  
    print(f"In testing, totally [{Y_test.shape[0]}] clip: ") 
    print(f">> number of class 1: [{np.sum((Y_test==[0,1]).all(axis=1))}]")
    print(f"Training...")
    learner.fit(X_train,Y_train) 
    print(f"Finished training") 
    print(short_line) 
    Y_train_h = learner.predict(X_train) 
    Y_test_h = learner.predict(X_test) 
    Y_h = learner.predict(X) 
    mis_ind = np.arange(len(clip_list))[(Y_h!=Y).all(axis=1)] 
    mislabeled = [clip_list[i] for i in mis_ind]
    print(f"Train accuracy is: [{learner.score(X_train,Y_train)}]")
    print(f"Test accuracy is: [{learner.score(X_test,Y_test)}]") 
    print(f"Mislabeled clips: [{len(mislabeled)}]") 
    
    return mislabeled_clips
    

# ====================================== end of objects and methods ====================================

def main(): 
    while(True): # keeps prompting for and train on data 
        print(long_line)
        clip_list = Prompt_for_data() 
        if len(clip_list)==0: 
            break  
        random.shuffle(clip_list) # randomized training is mandatory !!!!!
        print(f"Number of clips found: [{len(clip_list)}]") 
        learner = MLPClassifier(hidden_layer_sizes=mlp_prms['hidden'],solver='lbfgs',random_state=0)        
        # Y_train_h = learner.predict(X_train) 
        # Y_test_h = learner.predict(X_test) 
        mislabeled = Train_and_test_learner(learner,clip_list,test_ratio=mlp_prms['test_ratio'])
        
        
        
        # change this part 
        # msg,mislabeled = Train_and_test_learner(learner=learner, X=X, Y=Y, test_ratio=mlp_prms['test_ratio']) 
        # print(f"Printing out test result: ") 
        # print(short_line)
        # print(msg) 
        
        
        file_path = prompt_for_save_file(dir_path='mislabeled', f_format='.pkl') 
        if file_path==None: 
            continue
        with open(file_path, 'wb') as f: 
            pickle.dump(mislabeled, f) 
        print(f"File saved as {file_path}") 
        continue
        # ======================================================================================
    return 



# ====================================== user prompts =============================================== 


if __name__ == '__main__': 
    main() 
    exit(0) 