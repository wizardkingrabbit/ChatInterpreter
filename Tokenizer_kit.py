#!/usr/bin/python
# -*- coding: utf-8 -*-
# this module defines tokenizing tools for us to use in leaner modules. 
# typically tokanizing take a long string, returns a collection of tokens
# thus any processing to get the long string should be done outside the tokenizing functions

import os 
import numpy as np 
import nltk 
from nltk.corpus import stopwords 
from collections import defaultdict 
import re 

nltk_stop_words = set(stopwords.words('english'))

# =============================== small methods/tools ==================================================================== 

def Default_word_filter(word:str) -> bool: 
    ''' by default, every word is counted'''
    return True 

def Default_word_modifier(word:str) -> str: 
    ''' by default, words are not modified'''
    return word 

def BOW_to_str(bow:dict, freq_order=True, top_k=-1, indent='') -> str: 
    ''' This method turns a bow into user-friendly string for printing 
        printing order can be by frequency, or by a-b-c if by freq is false 
        the number of word to print out can be specified by top k 
        default, all words are printed out, with the specified indent'''
    to_return = '' 
    if freq_order:
        items = sorted(bow.items(), key=lambda x: x[1], reverse=True) 
    else: 
        items = sorted(bow.items(), key=lambda x: x[0]) 
        
    for word, freq in items: 
        to_return += indent 
        to_return += word 
        to_return += ' -> ' 
        to_return += str(freq) 
        to_return += os.linesep 
        top_k -= 1 
        if top_k == 0: 
            break 
    return to_return
    

# ================================= methods to make long string from collections =======================


def Concatenate_str_list(str_list:list, random_order=False, splitter=os.linesep) -> str: 
    ''' This function take a list of string, concatenate them into a long string 
        random order means items are concatenated in random order 
        splitter is what to put in between items in the long string''' 
    if not random_order: 
        to_return = splitter.join(str_list) 
        return to_return
    else: 
        index_arr = np.arange(len(str_list), dtype=int)
        index_arr = np.random.shuffle(index_arr) 
        to_return = '' 
        for i in index_arr: 
            assert type(str_list[i]) == str, 'concatenating non string items' 
            to_return += str_list[i] 
            to_return += splitter  
        return to_return 
    
    

# =============================== this part is for methods long strings into lists ==============================================

def Simple_tokenizer(long_string:str, remove_stop_words=True, stop_words=nltk_stop_words, case_sensitive=False) -> list: 
    ''' This module tokenize a long string, tokenize by white space, retaining marks such as question marks
        parameter options are whether to remove stop_words (from nltk module or specified otherwise), 
        and whether it is case sensitive 
        return value is a list of tokens made'''
    if not case_sensitive: 
        long_string = long_string.lower() 
    tokens = nltk.tokenize.word_tokenize(long_string) 
    if remove_stop_words: 
        to_return = [w for w in tokens if not(w in nltk_stop_words)] 
    else: 
        to_return = list(tokens) 
    return to_return 
    


# ================================ methods for processing list of tokens into BOW (sets, dicts, etc) ============================

def List_to_bow(token_list:list, filter_func=Default_word_filter, modifier_func=Default_word_modifier, n_gram=1, connector=''): 
    ''' takes a list of tokens, make them into a bag of words based on some conditions 
        n grams will be conted, default is 1, they will be connected with connector string specified
        first, the raw word has to pass a filter, by default, all are passed 
        second, the raw word is modified, by default, they remain the same 
        ''' 
    to_return = defaultdict(int) 
    ngram_window = [] 
    for token in token_list: 
        assert type(token) == str, 'passed token not string' 
        if not filter_func(token): 
            continue 
        ngram_window.append(modifier_func(token)) 
        if len(ngram_window) < n_gram: 
            continue
        if len(ngram_window) > n_gram: 
            ngram_window.pop(0) 
        for i in range(len(ngram_window)): 
            word = connector.join(ngram_window[:i+1]) 
            to_return[modifier_func(word)] += 1 
            

    while(len(ngram_window) > 0): 
        ngram_window.pop(0)
        for i in range(len(ngram_window)): 
            word = connector.join(ngram_window[:i+1]) 
            to_return[modifier_func(word)] += 1 

    return to_return 

    