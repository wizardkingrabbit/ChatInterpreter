#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import os 
import json 
import pickle 
import copy 
import sys 
import math 
import time 


long_line = '================================================================='
short_line = '---------------------------------------------------------------'
prompt_err_msg = 'invalid value entered, try again'



def prompt_for_int(message:str, min_v=None, max_v=None) -> int: # prompt for int with message, within min and max 
    ''' prompt for integer input with passed message and do all error checking,
        also check for min and max value range, they are inclusive'''
    while(True): 
        ans = input(message) 
        try: 
            ans = int(ans) 
            if min_v!=None: 
                assert ans>=min_v
            if max_v!=None: 
                assert ans<=max_v
            break
        except: 
            print('invalid value entered, try again') 
            continue 
        
    return ans 


def prompt_for_float(message:str, min_v=None, max_v=None) -> float: # prompt for a float with passed message between min and max 
    ''' prompt for float input with passed message and do all error checking,
        also check for min and max value range, they are inclusive'''
    while(True): 
        ans = input(message) 
        try: 
            ans = float(ans) 
            if min_v!=None: 
                assert ans>=min_v
            if max_v!=None: 
                assert ans<=max_v
            break
        except: 
            print('invalid value entered, try again') 
            continue 
        
    return ans 


def prompt_for_str(message:str, options={}) -> str: # prompt for a string, if there are options, check if within options
    ''' prompt for a string and check if it is in the options, if options not specified, if is returned directly''' 
    while(True): 
        ans = input(message) 
        if (len(options)) and not(ans in options): 
            print('invalid value entered, try again')
            continue
        else: 
            break 
        
    return ans 


def prompt_for_file_folder(message:str, exit_conds={}) -> str: # prompt for file path or exit conditions special str, check error  
    ''' prompt for either file or folder, returns path, type (folder/file)''' 
    while(True):
        file_path = input(message) 
        if file_path in exit_conds:
            return file_path, "exit_conds"
        elif os.path.isdir(file_path):
            return file_path, "folder"
        elif os.path.isfile(file_path):
            return file_path, "file"
        else:
            print("invalid path, please enter again")


def prompt_for_file(message:str, exit_conds={}) -> str: # prompt for file path or exit conditions special str, check error
    ''' prompt for a file path, exit upon entering exit condition string, do checking for validity''' 
    while(True):
        file_path = input(message) 
        if file_path in exit_conds: 
            return file_path 
        elif not os.path.isfile(file_path): 
            print('file path entered invalid, try again') 
            continue 
        else: 
            return file_path


def prompt_for_dir(message:str, exit_conds={}) -> str: # prompt for dir path or exit when special str passed, check error 
    ''' Takes a [message] string, set of str as exit conditions 
        prompt for a directory path using message 
        if one of the exit conds str is entered, returns that str 
        keeps prompting until valid path or exit cond 
        returns path once a valid path is entered''' 
    while(True): 
        dir_path = input(message) 
        if dir_path in exit_conds: 
            return dir_path 
        elif not os.path.isdir(dir_path): 
            print(f"dir path entered invalid, try again") 
            continue 
        else: 
            return dir_path 
        
            
def prompt_for_save_file(dir_path:str, f_format:str, ) -> str: # do a series of prompts for a path to save file
    ''' Takes a directory path, a format string for file 
        prompt for desired file name within that dir, return the file path
        if user do not want to save file, return None'''
    ans = prompt_for_str('save file? (y/n): ', options={'y','n'}) 
    if ans=='n': 
        return None 
    while(True): 
        print(f"file will be stored as ./{dir_path}/<YOUR FILE NAME>{f_format}")
        ans = prompt_for_str(f"Enter file name (WITHOUT {f_format}): ") 
        ans+=f_format
        file_path = os.path.join(dir_path, ans) 
        ans = prompt_for_str(f"file will be stored as {file_path}, are you sure? (y/n): ", options={'y','n'}) 
        if ans=='y': 
            break 
        else: 
            continue 
    return file_path 


def time_to_str(time:float) -> str: # turn a float time into hh:mm:ss formated string 
    ''' return a formated str of time in h:m:s'''
    time = int(time) 
    hours = time // 3600 
    minutes = (time % 3600) // 60 
    seconds = time % 60 
    
    return (f'{hours}:{minutes}:{seconds}') 

    
def Clip_chat_filter(chat:str, context:list) -> bool: 
    ''' Takes a chat message, determine if it is counted as a valid chat with context list 
        context is a list of chats that contains that chat, 
        returned value is bool''' 
        
    return True  


# print out a progress bar to be used in for loop
def Print_progress(i:int, n:int, object=None, message='') -> bool: 
    ''' Call this funciton in the for loop without any other print statements 
        the object is just for when it is called in list comprehension 
        only the first 30 characters in message will be used''' 
    message = message[:30]
    if i==n-1: 
        time.sleep(0.15)
        sys.stdout.write('\r' +' '*170 + '\r') 
        return object 
    progress = float(i+1)/n 
    progress = math.ceil(progress/0.01)
    to_print='\r Progress: [' 
    to_print += progress*'+' 
    to_print += (100-progress)*' ' 
    to_print += ']' 
    to_print += message 
    to_print += '\r'
    sys.stdout.write(to_print) 
    # time.sleep(0.001)
    return object  



# Funciton for random shuffle so we get a consistent result 
def Default_shuffle_func() -> float: 
    return 0.314