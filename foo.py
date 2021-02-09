from Utilities import * 
import os 
import pickle 
from Clip import *
from Tokenizer_kit import *


with open('clip_data/TeosGame[2].pkl', 'rb') as f: 
    data = pickle.load(f) 
    

print(len(data)) 
print(type(data))
sample = data[20] 
# print(sample) 
print(type(sample)) 

long_string = Concatenate_str_list(sample.chats) 
print(long_line) 
print('this is the long string: ') 
print(long_string) 

token_list = Simple_tokenizer(long_string) 

bow = List_to_bow(token_list, n_gram=2, connector='-') 

print(long_line) 
print('this is bow: ') 
print(BOW_to_str(bow, indent=' ', top_k=10)) 
