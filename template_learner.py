#!/usr/bin/python
# -*- coding: utf-8 -*-
# this is the template learner, do not change this file but make copies and name them accordingly


# import all you need
import os 
from Utilities import * 
from Clip import * 
from Tokenizer_kit import * 
import pickle

import warnings 
warnings.filterwarnings(action='ignore') 

# ======================================= objects and methods =======================================

# ====================================== data loader =================================================
def Load_clips_from_file(file_path:str) -> list: # load clip list from a passed file path 
    ''' load the clip list from given file path, 
        return empty list if a file is not formated as a list of clips
        error if a file is not pickle file or path not valid''' 
    with open(file_path, 'rb') as f: 
        data = pickle.load(f) 
        
    if (type(data) != list) or (len(data) < 1) or (type(data[0]) != clip_it): 
        return list() 
    else: 
        return data 
    

def Directory_load(dir_path:str) -> list: # load clip data from a directory
    ''' load all clip data from a directory into one list, 
        error if path not valid or one file is not formated as pkl
        return the resulting list'''
    to_return = list() 
    for file_path in os.listdir(dir_path): 
        to_return += Load_clips_from_file(file_path=os.path.join(dir_path,file_path)) 
    return to_return 


def Sequential_load() -> list: # keeps prompting for files to load
    ''' keeps prompting for files to load into clip list 
        returns empty list if one want to exit ''' 
    to_return = list() 
    while(True): 
        print(f"Enter pickle file path (with .pkl), 'fin' to finish, 'e' to exit: ")
        file_path = prompt_for_file(f"Enter here: ", exit_conds={'fin','e'}) 
        if (file_path=='fin') and (len(to_return)>0): 
            return to_return 
        elif (file_path=='fin') and (len(to_return)==0): 
            print(f"you have to enter at least one file") 
            continue 
        elif file_path=='e': 
            return list() 
        else: 
            try: 
                to_return += Load_clips_from_file(file_path=file_path) 
            except: 
                print(f"Entered file invalid") 
                continue 
    return to_return
                 
            
def Prompt_for_data() -> list: # prompt for and load data 
    ''' This function prompt for data loading mode and load them in with special prompts 
        function is called in main to return the final list of clips 
        returns empty list if user want to exit''' 
    to_return = list() 
    while(True): 
        print(f"Do you want to train on folder 'f' or enter files sequentially 's' or exit 'e'?: ")
        ans = prompt_for_str(f"Enter here: ", options={'s','f','e'}) 
        if ans=='e': # exit
            break 
        elif ans=='f': # load from folder
            dir_path = prompt_for_dir(f"Enter folder path, 'e' to exit: ", exit_conds={'e'}) 
            if dir_path=='e': continue
            to_return =  Directory_load(dir_path=dir_path) 
            break 
        elif ans=='s':  # load sequentially
            clip_list = Sequential_load() 
            if len(clip_list)==0: continue 
            to_return = clip_list 
            break 
    return to_return
            

# ===================================== learner methods ============================================

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
            continue 
            
        
        
    return 



# ====================================== user prompts =============================================== 


if __name__ == '__main__': 
    main() 
    exit(0) 