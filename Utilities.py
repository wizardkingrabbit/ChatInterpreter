#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import os 
import json 
import pickle 
import copy 


long_line = '================================================================='
short_line = '---------------------------------------------------------------'
prompt_err_msg = 'invalid value entered, try again'



def prompt_for_int(message:str, min_v=None, max_v=None) -> int: 
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


def prompt_for_float(message:str, min_v=None, max_v=None) -> float: 
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

def prompt_for_str(message:str, options={}) -> str: 
    ''' prompt for a string and check if it is in the options, if options not specified, if is returned directly''' 
    while(True): 
        ans = input(message) 
        if (len(options)) and not(ans in options): 
            print('invalid value entered, try again')
            continue
        else: 
            break 
        
    return ans 


def prompt_for_file(message:str, exit_conds={}) -> str: 
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

def time_to_str(time:float) -> str: 
    ''' return a formated str of time in h:m:s'''
    time = int(time) 
    hours = time // 3600 
    minutes = (time % 3600) // 60 
    seconds = time % 60 
    
    return (f'{hours}:{minutes}:{seconds}') 

    


