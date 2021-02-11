from Utilities import * 
import os 
import pickle 
from Clip import *
from Tokenizer_kit import *

import gensim
from gensim.models import Word2Vec



pkl_file = (open("chat_words/TeosGame.pkl", "rb")) 
data = pickle.load(pkl_file) 

print(type(data)) 
print(len(data))

sample = data[1000:1020] 
text = Concatenate_str_list(sample) 
print(text)

