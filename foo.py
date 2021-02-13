#!/usr/bin/python
# -*- coding: utf-8 -*-

from Utilities import * 
import os 
import pickle 
from Clip import *
from Tokenizer_kit import *

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors 


def Compare_two_words(w1:str, w2:str, w_vector, top_n=10): 
    print(long_line)
    print(f"[{w1}]:[{w2}] has similarity {w_vector.similarity(w1, w2)}") 
    if not(top_n): 
        return
    print(f">> words similar to [{w1}] are: ") 
    for w,v in w_vector.most_similar(w1, topn=top_n): 
        print(f">>>> {w}: {v}") 
    
    print(short_line)
    print(f">> words similar to [{w2}] are:") 
    for w,v in w_vector.most_similar(w2, topn=top_n): 
        print(f">>>> {w}: {v}") 
        

wv = KeyedVectors.load('word_vectors/TeosGame_wv.kv') 


# Compare_two_words('teo', 'pog', wv, top_n=0) 
# Compare_two_words('lukas', 'nice', wv, top_n=0) 
# Compare_two_words('paddy', 'nice', wv, top_n=0) 
# Compare_two_words('sammy', 'nice', wv, top_n=0)
# # Compare_two_words('paddy', 'teo', wv)  
# Compare_two_words('katie', 'nice', wv, top_n=0) 

for name in ['teo', 'lukas', 'katie', 'sammy', 'alex', 'paddy', 'butt']: 
    Compare_two_words(name, 'nice', wv, top_n=0) 
    Compare_two_words(name, 'pog', wv, top_n=0) 
    Compare_two_words(name, 'lul', wv, top_n=0) 
    Compare_two_words(name, 'kekw', wv, top_n=0) 
    Compare_two_words(name, 'pepelaugh', wv, top_n=0) 

# print(wv.closer_than('teo', 'kekw'))
