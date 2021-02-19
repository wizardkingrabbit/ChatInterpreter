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
import log_regression

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

# please be sure that you give it a valid path when using it
def add_filepath_to_set(the_path:str, is_file:bool, original_set):
    if (is_file):
        original_set.append(the_path)
    else:
        for entry in os.scandir(the_path):
            if (entry.path.endswith(".pkl") and entry.is_file()):
                original_set.append(entry.path)
    return original_set

# interpret a pkl file and extract its data into three lists
def add_clipdata_to_set(clip_list, text_list, y_list, pkl_path):
    the_file = open(pkl_path, 'rb')
    the_pkl = pickle.load(the_file)
    for clip in the_pkl:
        clip_list.append(clip)
        text_list.append(Concatenate_str_list(clip.chats))
        if (clip.get_label_binary() == 0):
            y_list.append(0)
        else:
            y_list.append(1)
    the_file.close()
    return clip_list, text_list, y_list


# ====================================== end of objects and methods ====================================

#def main(): 
#    while(True): # keeps prompting for and train on data 
#        print(long_line)
#        clip_list = Prompt_for_data()
#        if len(clip_list)==0: 
#            break  
#        # delete this part if you want
#        else: 
#            print(f"Number of clips found: [{len(clip_list)}]")  
#
#        # change this part 
#        msg,mislabeled = Train_and_test_learner(learner=None, X=clip_list, Y=None, test_ratio=0.2) 
#        print(f"Printing out test result: ") 
#        print(short_line)
#        print(msg) 
#        
#        file_path = prompt_for_save_file(dir_path='mislabeled', f_format='.pkl') 
#        if file_path==None: 
#            continue
#        with open(file_path, 'wb') as f: 
#            pickle.dump(mislabeled, f) 
#        print(f"File saved as {file_path}") 
#        continue
#
#    return 

def main(method = None): 
    # main function, a sequence of supportive methods defined above 
    # see specifications in learner_output.txt \
    # one good practice is to keep indent within a function no more than 3
    # if more loop like structures are needed, another defined method is recommended

    # define method
    if method == None:
        method = prompt_for_str("which method to use? (linear/RNN)", {"linear", "RNN"})
    #define training set
    filepath = []
    text = []
    Y = []
    all_clip = []
    file_or_folder, _type = prompt_for_file_folder("enter a path to a file or a folder to add that to the training set, enter e to exit", {"e"})
    while(file_or_folder != "e"):
        filepath = add_filepath_to_set(file_or_folder, _type == "file", filepath)
        file_or_folder, _type = prompt_for_file_folder("enter a path to a file or a folder to add that to the training set, enter e to exit", {"e"})
    for filename in filepath:
        all_clip, text, Y = add_clipdata_to_set(all_clip, text, Y, filename)
    training_size = len(Y)
    #define validation set
    filepath = []
    file_or_folder, _type = prompt_for_file_folder("enter a path to a file or a folder to add that to the validation set, enter e to exit", {"e"})
    while(file_or_folder != "e"):
        filepath = add_filepath_to_set(file_or_folder, _type == "file", filepath)
        file_or_folder, _type = prompt_for_file_folder("enter a path to a file or a folder to add that to the validation set, enter e to exit", {"e"})
    for filename in filepath:
        all_clip, text, Y = add_clipdata_to_set(all_clip, text, Y, filename)
    validation_size = len(Y) - training_size
    #train the model
    if (method == "linear"):
        classifier, t_err, v_err, t_msg, v_msg = log_regression.main(text, Y, training_size, validation_size)
    if (method == "RNN"):
        # call RNN method here
        # classifier, t_err, v_err, t_msg, v_msg = ...
        pass
    print(t_msg)
    print(v_msg)
    #save the mislabeled
    if (prompt_for_str("Do you want to save the mislabeled clips? (y/n) ") == "y"):
        if not os.path.isdir("/mislabeled"):
            os.mkdir("/mislabeled")
        file_prefix = prompt_for_str("Please name the prefix of saved files: ")
        # making mislabeled file for training errors
        err_list = list()
        for err_id in t_err:
            err_list.append(all_clip[err_id])
        new_file_path = 'mislabeled/' + file_prefix + '_mislabeled_train.pkl' 
        with open(new_file_path, 'wb') as f: 
            pickle.dump(err_list, f)
        # making mislabeled file for validation errors
        err_list = list()
        for err_id in v_err:
            err_list.append(all_clip[err_id + training_size])
        new_file_path = 'mislabeled/' + file_prefix + '_mislabeled_validation.pkl' 
        with open(new_file_path, 'wb') as f: 
            pickle.dump(err_list, f)
    while (input("Do you want to test this classifier on any unlabled clip data? (y/n)") == "y"):
        pass # to be done
    return 

# ====================================== user prompts =============================================== 


if __name__ == '__main__': 
    main() 
    exit(0) 