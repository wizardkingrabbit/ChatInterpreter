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

def main_alter(method = None): 
    # main function, a sequence of supportive methods defined above 
    # see specifications in learner_output.txt \
    # one good practice is to keep indent within a function no more than 3
    # if more loop like structures are needed, another defined method is recommended

    # define method
    if method == None:
        method = prompt_for_str("which method to use? (linear/RNN)", {"linear", "RNN"})
    #define training set
    all_clip = []
    filepath = []
    file_or_folder, _type = prompt_for_file("enter a path to a file or a folder to add that to the training set, enter e to exit", {"e"}, True)
    while(file_or_folder != "e"):
        filepath = add_to_set(file_or_folder, _type == "file", filepath)
        file_or_folder, _type = prompt_for_file("enter a path to a file or a folder to add that to the training set, enter e to exit", {"e"}, True)
    text = []
    Y = []
    for filename in filepath:
        the_file = open(filename, 'rb')
        the_pkl = pickle.load(the_file)
        for clip in the_pkl:
            all_clip.append(clip)
            text.append(Concatenate_str_list(clip.chats))
            if (clip.get_label_binary() == 0):
                Y.append(0)
            else:
                Y.append(1)
        the_file.close()
    training_size = len(Y)
    #define validation set
    file_or_folder, _type = prompt_for_file("enter a path to a file or a folder to add that to the validation set, enter e to exit", {"e"}, True)
    while(file_or_folder != "e"):
        filepath = add_to_set(file_or_folder, _type == "file", filepath)
        file_or_folder, _type = prompt_for_file("enter a path to a file or a folder to add that to the validation set, enter e to exit", {"e"}, True)
    for filename in filepath:
        the_file = open(filename, 'rb')
        the_pkl = pickle.load(the_file)
        for clip in the_pkl:
            all_clip.append(clip)
            text.append(Concatenate_str_list(clip.chats))
            if (clip.get_label_binary() == 0):
                Y.append(0)
            else:
                Y.append(1)
        the_file.close()
    validation_size = len(Y) - training_size
    #train the model
    if (method == "linear"):
        classifier, t_err, v_err = log_regression.main(text, Y, training_size, validation_size)
    #save the mislabeled
    if (prompt_for_str("Do you want to save the mislabeled clips? (n/y) ") == "y"):
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
    return 

# ====================================== user prompts =============================================== 


if __name__ == '__main__': 
    main() 
    exit(0) 