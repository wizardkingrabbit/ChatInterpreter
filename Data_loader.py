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

'''
================================== module description =====================================
Data loader is here to provide methods for learner to load data, but calling this module will 
prompt for a clip pkl file to inspect. This is intended for later checking mislabeled data. 
Same thing can be done using labeler, but with more tedious procedures, thus this module. 
'''
# ====================================== data loader methods =========================================
# load clip list from a passed file path
def Load_clips_from_file(file_path:str) -> list:  
    ''' load the clip list from given file path, 
        return empty list if a file is not formated as a list of clips
        error if a file is not pickle file or path not valid''' 
    with open(file_path, 'rb') as f: 
        data = pickle.load(f) 
        
    if (type(data) != list) or (len(data) < 1): 
        return list() 
    else: 
        return data 
    

# load clip data from a directory
def Directory_load(dir_path:str) -> list: # load clip data from a directory
    ''' load all clip data from a directory into one list, 
        error if path not valid or one file is not formated as pkl
        return the resulting list'''
    to_return = list() 
    for file_path in os.listdir(dir_path): 
        to_return += Load_clips_from_file(file_path=os.path.join(dir_path,file_path)) 
    return to_return 



# keeps prompting for files to load
def Sequential_load() -> list:
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
                 

# prompt for and load data 
def Prompt_for_data() -> list: 
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
            

# load vod json file to comment list
def Load_json(file_path=None) -> list: 
    if file_path==None: 
        file_path = prompt_for_file(f"Enter json file path") 
    with open(file_path, encoding='utf-8') as f: 
        data = json.load(f) 
    return data['comments'] 


# inspect the clips from passed clip list on terminal prompts
def Inspect_data(data): 
    for i,clip in enumerate(data): 
        clip = clip.copy()
        print(short_line) 
        print(f"Clip of [{i+1}/{len(data)}]: ") 
        print(clip) 
        ans = prompt_for_str(f"Next one? (hit enter): ", options={'',}) 

        

        
    
# ================================= end of loader methods ======================================
def main(): 
    while(True): 
        print(long_line) 
        print(f"Enter the file you want to inspect (with .pkl), 'e' to exit")
        file_path = prompt_for_file(f"Enter here: ", exit_conds={'e'}) 
        if file_path=='e': break  
        try: 
            data = Load_clips_from_file(file_path) 
        except: 
            print(f"Entered file invalid") 
            continue 
        Inspect_data(data) 
        continue
    return 
    
    
if __name__=='__main__': 
    main() 
    exit(0)