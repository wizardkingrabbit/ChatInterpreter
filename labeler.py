#!/usr/bin/python
# -*- coding: utf-8 -*-
from Utilities import * 
import numpy as np 
import os 
import json 
import pickle 
import copy 

''' This module prompt the user to enter clip pickle files and label them by hand
    users will also enter clip span duration, but that is optional''' 
    
    
if __name__ == '__main__': 
    while(True):
        file_path = input('Enter pickle file path (WITH .pkl, enter exit to exit): ') 
        
        if type(file_path) != str: 
            print('invalid value entered, try again') 
            continue 
        elif file_path == 'exit': 
            exit(0) 
        elif not os.path.isfile(file_path): 
            print('file path entered invalid, try again') 
            continue 
        else: 
            break 
        
    with open(file_path, 'rb') as f: 
        clip_list = pickle.load(f) 
    
    video_id = clip_list[0].video_id 
    labeled_list = list()
    N = len(clip_list)
    
    for i in range(len(clip_list)): 
        clip = clip_list[i].copy()
        if not clip.is_valid(): 
            print(f'found one invalid clip at index {i}')
            continue 
        
        print(long_line) 
        print(f'labeling {i+1}/{N} clips')
        print(f'video id is {video_id}') 
        print(f'chat duration is {clip.chat_duration_to_str()}') 
        print(f'there are {len(clip)} chats') 
        while(True): 
            ans = input('Do you want to see the chat? (y/n/e, e for exit): ') 
            if ans in {'y', 'n', 'e'}: 
                break 
            else: 
                print('invalid value entered') 
                continue 
            
        if ans == 'e': 
            exit(0)
        elif ans == 'y': 
            print('printing chat') 
            print(short_line) 
            print(clip) 
        elif ans=='n': 
            print(short_line)
            print(f'Available labels are {clip.label_info_to_str()}')
            print(f'current label of the clip is [{clip.get_label()}]')
            print(f'current binary label for the clip is [{clip.get_label_binary()}]')
        ans = prompt_for_int('Enter a label index in int: ',min_v=0, max_v=len(clip.label_info())-1)  
        
        clip.set_label(ans) 
             
        
        while(True): 
            print(short_line)
            ans = input('Do you want ot enter a span duration? (y/n/i, i for more info): ')
            if not ans in {'y', 'n', 'i'}: 
                print('invalid value entered, try again') 
                continue 
            elif ans == 'i': 
                print(short_line)
                print('span duration is how many chat duration do we go back to get the clip') 
                print('for example, if chat peek has time duration of 6.3 seconds, and span duration is 5 (default)') 
                print('we assume the content start from 6.3x5=31.5 seconds before the start of chat speed peek') 
                continue
            else:
                break 
            
        if ans == 'y': 
            span_ans = prompt_for_float('Enter duration in float: ', min_v=0.0)      
            clip.set_span_duration(span_ans) 
            
        labeled_list.append(clip) 
        
    while(True): 
        ans = input('Save file? (y/n): ') 
        if not(ans in {'y','n'}): 
            print('invalid value entered, try again') 
            continue 
        else: 
            break 
    
    if ans=='y': 
        new_file_path = file_path[:-4:] 
        new_file_path += '_labeled' 
        new_file_path += '.pkl' 
        
        with open(new_file_path, 'wb') as f: 
            pickle.dump(labeled_list, f) 
            
        print(f'Saved file as {new_file_path}')
    else: 
        print('exiting')
        
        
    exit(0)
        
                
        
        
        
    