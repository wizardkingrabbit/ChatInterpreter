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

import warnings 
warnings.filterwarnings(action='ignore') 

# ======================================= objects and methods =======================================


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