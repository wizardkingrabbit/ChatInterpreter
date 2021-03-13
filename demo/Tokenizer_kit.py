#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
this module defines tokenizing tools for us to use in leaner modules. 
typically tokanizing take a long string, returns a collection of tokens
thus any processing to get the long string should be done outside the tokenizing functions
'''

import os 
import numpy as np 
import nltk 
from nltk.corpus import stopwords 
from collections import defaultdict 
import re 

nltk_stop_words = set(stopwords.words('english'))

# =============================== small methods/tools ==========================================================================

# filter function for List_to_BOW
def Default_word_filter(word:str) -> bool: 
    ''' by default, every word is counted'''
    return True 

# modifier function for List_to_BOW
def Default_word_modifier(word:str) -> str: 
    ''' by default, words are not modified'''
    return word 

# converts a bag of words to a string for printing
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
    

'''============================================== methods to make long string from collections ================================================'''
# concate a list of string into a long string 
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
    

'''=============================== this part is for methods long strings into lists ============================================================'''
# simple tokenizer using nltk
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
    

# default tokenizer that separate by space 
def Default_tokenizer(long_string:str, stopwords={}, min_len=1) -> list: 
    ''' this default tokenizer split sentence by white space, 
        remove stop words from passed set, 
        remove any token that is shorter than minimum length 
        return the token list ''' 
    return [i for i in long_string.split() if not((i in stopwords) or (len(i)<min_len))] 


#--------------------- Embedding tokenizer -------------------------------------------
embd_stop_words = set(nltk_stop_words) 

# modify word to account for word variations
def Embedding_word_modifier(word:str, stop_words = embd_stop_words) -> str: 
    ''' Takes a word, check for varies conditions, make modifications, return result, None if in stop words''' 
    
    # match F, special case 
    F = re.compile("^\s[Ff]\s$") 
    if re.match(F, word): 
        return "F" 
    # other than special cases, remove stop words
    if word in stop_words: 
        return None
    # match question marks ???
    q_marks = re.compile("^\?{2,}$") 
    if re.match(q_marks, word): 
        return "???" 
    # match exclmtn marks !!!
    exclmtn_marks = re.compile("^!{2,}$") 
    if re.match(exclmtn_marks, word): 
        return "!!!" 
    # match ??!?!??!? 
    q_exc_marks = re.compile("^[!?]{2,}$") 
    if re.match(q_exc_marks, word): 
        return "!?" 
    # match variations of pog
    pog = re.compile("^p+o+g+$") 
    if re.match(pog, word): 
        return "pog" 
    # match variations of nice
    nice = re.compile("^n+i+c+e+u*|n+a+i+s+u+$") 
    if re.match(nice, word): 
        return "nice" 
    # match variations of noice 
    noice = re.compile("^n+o+i+c+e+$") 
    if re.match(noice, word): 
        return "noice" 
    # match variations of haha 
    haha = re.compile("^(ha){2,}h?$|h{3,}$") 
    if re.match(haha, word): 
        return "haha" 
    # match variations of lol 
    lol = re.compile("^l+o+l+$") 
    if re.match(lol, word): 
        return "lol" 
    # match variations of lul 
    lul = re.compile("^l+u+l+$") 
    if re.match(lul, word): 
        return "lul" 
    # match variations of lmao
    lmao = re.compile("^lmf?ao+$") 
    if re.match(lmao, word): 
        return "lmao" 
    # match variations of yes 
    yes = re.compile("^y+e+s+$") 
    if re.match(yes, word): 
        return "yes" 
    # match variations of noo 
    noo = re.compile("^n+o{2,}$") 
    if re.match(noo, word): 
        return "noo" 
    # match variations of no 
    no = re.compile("^(no){2,}$") 
    if re.match(no, word): 
        return "no" 
    # match variations of yeah 
    yeah = re.compile("^y+e+a+h*$|^ya$|^ye+$") 
    if re.match(yeah, word): 
        return "yeah" 
    # match variations of ree 
    ree = re.compile("^r+e+$") 
    if re.match(ree, word): 
        return "ree" 
    # match variations of oof 
    oof = re.compile("^o{2,}f+$") 
    if re.match(oof, word): 
        return "oof" 
    # match variations of pogu 
    pogu = re.compile("^p+o+g+u+$") 
    if re.match(pogu, word): 
        return "pogu" 
    # xd 
    xd = re.compile("^xd+$") 
    if re.match(xd, word): 
        return "xd" 
    # ez 
    ez = re.compile("^e+z+$") 
    if re.match(ez, word): 
        return "ez" 
    # money 
    money = re.compile("^mo+ne+y+$") 
    if re.match(money, word): 
        return "money" 
    
    return word 
    

# tokenize a sentence for embedding training, 
def Embedding_tokenize(sentence:str, word_filter=Embedding_word_modifier, case_sensitive=False) -> list: 
    ''' Tokenization customized to embedding, takes a sentence and returns a list of tokens''' 
    # chech case sensitivity 
    if not case_sensitive: 
        sentence = sentence.lower() 
    # make token pattern and match them
    token_pattern = "\s[Ff]\s|[?!]{2,}|[:;]\)|[^\s',@.():?!]{2,}|\w:|:\w\s" 
    token_pattern = re.compile(token_pattern) 
    raw_tokens = re.findall(token_pattern, sentence) 
    
    # put each token through a modifier, which also check for validity 
    to_return = list()
    for token in raw_tokens: 
        modified = Embedding_word_modifier(word=token) 
        if modified!=None: 
            to_return.append(modified) 
    return to_return 


'''================================ methods for processing list of tokens into BOW (sets, dicts, etc) ============================================='''

# converts list of tokens to bag of words
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

 
   
