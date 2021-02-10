from Utilities import * 
import os 
import pickle 
from Clip import *
from Tokenizer_kit import *

stop_words = set(nltk_stop_words) 

stop_words = set()


print(type(nltk_stop_words))

for word in nltk_stop_words: 
    print(' ' + word)
