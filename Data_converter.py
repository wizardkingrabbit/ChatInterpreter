from Tokenizer_kit import * 
import numpy as np 
from Embedding import *
from Utilities import *
from Data_loader import *
from collections import defaultdict 
from Clip import *
import gensim 
import random 


'''
================================== Module Description =====================================================
This module is implemented to be a collection of methods for converting one data structure into another
For example: turn a corpus of sentences into one-hot vector mapping of each tokens
calling the main function in this module will prompt in terminal to test module methods on data 
Other modules can import this module to use its methods 
'''

# Turns a clip object into a tuple of (chat list, label int)
def Clip_to_tuple(clip:clip_it, binary=True) -> tuple: 
    if binary: 
        label = clip.get_label_binary()**2 
    else: 
        label = clip.get_label 
    return (clip.chats, clip.get_label_binary)


# Turn a list of clips into a list of tuples, eahc is (chat list, label) 
def Clip_list_to_tuples(clip_list:list, binary=True, randomize=False) -> list: 
    ''' This function takes a list of clips, extract their information
        turn each clip into a tuple of (chat list, label int) 
        label int is determined by binary parameter 
        randomize will randomize the returned list for better learner training''' 
    to_return = [Clip_to_tuple(i,binary) for i in clip_list] 
    if randomize: 
        random.shuffle(to_return)
    return to_return


# takes a vector, returns a dict that maps tokens to one hot vectors
def One_hot_vectorizer(corpus:list, tokenizer=Default_tokenizer) -> dict: 
    ''' Take a corpus as a list of strings, tokenize all of the strings using passed tokenizer 
        tokenizer should be a function that takes a string and return a list of tokens
        tokenizer function should do all the work (e.g. stop word checking, word modification)
        using the long list of tokens, make a one-hot vector dictionary {token:vector} 
        vectors will be numpy arrays''' 
    token_set = set() 
    token2vec = dict() 
    for sentence in corpus: 
        token_set.update(tokenizer(sentence)) 
    token_list = list(token_set) 
    vector_len = len(token_list)
    for i,token in enumerate(token_list): 
        vector = np.zeros(vector_len, dtype=np.double) 
        vector[i]=1.0 
        token2vec.update({token:vector}) 
    return token2vec 

    
# test One_hot_vectorizer on a list of clips 
def Test_ohv(clip_list:list, binary=True): 
    print(short_line) 
    if len(clip_list)==0: 
        print(f"You passed 0 clips") 
    print(f"Testing One_hot_vectorizer on [{len(clip_list)}] clips")
    data = Clip_list_to_tuples(clip_list, binary) 
    corpus = list() 
    for i in data: 
        corpus += i[0] 
    one_hot = One_hot_vectorizer(corpus) 
    print(f"Number of tokens: [{len(one_hot)}]") 
    sample = list(one_hot.values())[0] 
    print(f"Dimention of vectors: [{sample.shape}]") 
    while(True): 
        ans=prompt_for_str(f"Enter a token you want, 'e' to exit: ") 
        if ans=='e': break  
        vector = one_hot.get(ans,np.zeros(sample.shape, dtype=np.double))
        print(f"Vector for token [{ans}] is: ") 
        print(f"type: [{type(vector)}]") 
        print(f"With magnitude: [{np.sum(vector)}]")
        print(vector) 
    return 
    
     
    
# =========================================== main function ==================================================
        
def main(): 
    while(True): 
        print(long_line)
        print(f"Enter testing options: ") 
        print(f"[ohv]: to test one-hot vector") 
        print(f"[e]: exit")
        ans=prompt_for_str("Enter here: ", options={'ohv','e'}) 
        if ans=='e': break 
        print(f"Enter your testing data")
        clip_list = Prompt_for_data() 
        if ans=='ohv': 
            Test_ohv(clip_list) 
    return 



if __name__=='__main__': 
    main() 
    exit(0)