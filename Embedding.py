#!/usr/bin/python
# -*- coding: utf-8 -*-
# this is the template learner, do not change this file but make copies and name them accordingly


# import all you need
import os 
from Utilities import * 
from Clip import * 
from Tokenizer_kit import * 
from Clipper import * 
import json 
import gensim.models


import warnings 
warnings.filterwarnings(action='ignore')


# ======================================= objects and methods =======================================


def Process_vod(data): 
    ''' This function takes a data object from json file, process it into a list of chats 
        chats are ignored if they are outside specified start and end time''' 
        
    _,chat_array,t_stamps,_ = Twitch_Comment_to_data(data['comments'], chat_window=1) 
    assert len(chat_array) == len(t_stamps) 
    return (chat_array, t_stamps)
    
    
def Cut_ends(chat_array:np.array, t_stamps:np.array, start_time:float, end_time:float) -> list: 
    ''' This function cut ends of a chat list based on start time and end time''' 
    to_return = list() 
    for i in range(len(t_stamps)): 
        if start_time<t_stamps[i] and t_stamps[i]<end_time: 
            to_return.append(str(chat_array[i])) 
            
    return to_return 


def Prompt_for_start_time(max_v:int) -> float: 
    ''' prompt for start time with a specified maximum, message is fixed, float is returned'''
    while(True):
        start_time = prompt_for_int('Enter a time (in seconds) when greeting ends, -1 for more info: ', min_v=-1, max_v=max_v) 

        if start_time==-1: 
            print(short_line)
            print('This is for excluding the beginning of the vod where people come in and greet') 
            print('No default value, you can enter any value as long as it is within vod limit') 
        else: 
            break 
    start_time = float(start_time) 
    return start_time 
        
        
def Prompt_for_end_duration(max_v:int) -> float: 
    ''' prompt for an duration of ending, message is fixed, float is returned''' 
    while(True):    
        ending_duration = prompt_for_int('Enter expected ending duration (in seconds), -1 for more info: ', min_v=-1, max_v=max_v) 
        
        if ending_duration==-1: 
            print(short_line)
            print('This number is how long you expect ending and saying goodbye in the stream will be') 
            print('No default, you can enter any value as long as it is within vod limit') 
        else: 
            break 
    return float(ending_duration)


def Thread_chats(chat_list:list, block_size=100) -> list: 
    ''' This function thread together chat messages, every block of chat is threaded into one sentence
        returns a list of threaded sentences'''
    to_return = list() 
    i=0
    while (i < len(chat_list)): 
        i+=100 
        sentence = chat_list[i-100:i] 
        sentence = Concatenate_str_list(str_list=sentence, splitter=' ') 
        to_return.append(sentence) 
        
    return to_return
        
        

custom_stop_words = set(nltk_stop_words) 
def Embedding_tokenize(sentence:str) -> list: 
    ''' Tokenization customized to embedding, takes a sentence and returns a list of tokens''' 
    to_return = Simple_tokenizer(long_string=sentence, stop_words=custom_stop_words)
    return to_return


def Most_similar_to(word_vector, word:str, top_k:int) -> str: 
    ''' This function takes a word string, word vector object, and an int of how many to print out 
        returns a reader-friendly string to be printed out, or a special string when word is not in vocab''' 
    to_return = (short_line + os.linesep)
    to_return += f"{top_k} most similar words of [{word}] are: " + os.linesep  
    try: 
        for w,v in word_vector.wv.most_similar(word, topn=top_k): 
            to_return += f">>[{w}]: {v} {os.linesep}" 
    except KeyError: 
        return f"Word [{word}] not in vocabulary" 
    
    return to_return


def Compare_two_words(word_vector, w1:str, w2:str) -> str: 
    ''' Takes a word vector object and compute cosine similarity, return result as a str to print, 
        special string is returned if word not in vocab''' 
    try: 
        to_return = (short_line + os.linesep)
        to_return += (f"[{w1}]:[{w2}] has similarity {word_vector.wv.similarity(w1, w2)}") 
        return to_return 
    except KeyError: 
        return "One of the words is not in vocab" 


def Check_trained_model(word_vector): 
    ''' This is for main to call when user want to check a trained model''' 
    while(True): 
        print(short_line)
        print('Enter either one word to find most similar or two to find similarity, enter 0 to exit')
        sentence = prompt_for_str('Enter here: ') 
        if sentence=='0': 
            break 
        tokens = Embedding_tokenize(sentence=sentence) 
        
        if len(tokens)==1: 
            top_k = prompt_for_int('How many similar words do you want to see?: ', min_v=1) 
            word = tokens[0] 
            print(Most_similar_to(word_vector=word_vector, word=word, top_k=top_k)) 
        elif len(tokens)==2: 
            w1 = tokens[0] 
            w2 = tokens[1] 
            print(Compare_two_words(word_vector=word_vector, w1=w1, w2=w2)) 
        else: 
            print("Tokens in your sentences are: ") 
            print(tokens) 
            print("number of tokens is not valid") 
            continue
    return 

def Train_new_model() -> gensim.models.KeyedVectors: 
    ''' This is for main to call when user want to train a new model 
        returns the keyed vector object as trained result''' 
    first_run=True  
    model = gensim.models.Word2Vec(min_count=20, size=300, window=7, iter=5) 
    print('Training new word2vec object...')
    while(True): 
        print(short_line)
        if not first_run:
            print('Keep training')
        print("Enter json text file path (WITH .json, enter 'fin' to finish training, 'check' to check current model)")
        file_path = prompt_for_file('Enter here: ', exit_conds={'fin','check'}) 
        if file_path=='fin': 
            print(f"Finished training")
            break 
        elif file_path=='check': 
            print(long_line)
            Check_trained_model(word_vector=model) 
            continue
        try:
            with open(file_path, encoding='utf-8') as f: 
                data = json.load(f) 
            chat_array, t_stamps = Process_vod(data=data)  
        except: 
            print('file entered not valid') 
            continue
        
        end_time = t_stamps[-1] 
        start_time = Prompt_for_start_time(max_v=int(end_time)) 
        ending_duration = Prompt_for_end_duration(max_v=end_time) 
        end_time -= ending_duration 
        if end_time <= start_time: 
            print('Time interval invalid, enter info again') 
            continue 
        
        raw_chats = Cut_ends(chat_array=chat_array, t_stamps=t_stamps, start_time=start_time, end_time=end_time) 
        raw_chats = Thread_chats(chat_list=raw_chats) 
        chats_to_train = list() 
        for sentence in raw_chats:
            chats_to_train.append(Embedding_tokenize(sentence=sentence)) 
            
        print(f"Training...") 
        
        if(first_run): 
            model.build_vocab(chats_to_train) 
            first_run=False 
        else: 
            model.build_vocab(chats_to_train, update=True) 
            
        model.train(chats_to_train, total_examples=len(chats_to_train), epochs=model.epochs) 
        print(f"Finished training")
        continue
    
    return model.wv 
            

def Save_wv(word_vector): 
    ''' This function takes a keyed word vector and store is with a series of prompts'''
    ans = prompt_for_str('store word vector file? (y/n): ', options={'y','n'}) 
    if ans=='n': 
        return 
    print('file will be stored as ./word_vectors/<YOUR FILE NAME>.kv')
    while(True): 
        ans = prompt_for_str('Enter file name (WITHOUT .kv): ') 
        ans+='.kv' 
        kv_path = os.path.join('word_vectors', ans) 
        ans = prompt_for_str(f"file will be stored as {ans}, are you sure? (y/n)", options={'y','n'}) 
        if ans=='y': 
            break 
        else: 
            continue 
    print(f"Saving file as {kv_path}")
    word_vector.save(kv_path) 
    print('File saved') 
    return 


def Prompt_for_wv() -> gensim.models.KeyedVectors: 
    ''' This function is called by main to prompt for a keyed vector file 
        which is loaded and returned ''' 
    while(True): 
        file_path = prompt_for_file(f"Enter .kv file path (WITH .kv): ") 
        try:
            to_return = gensim.models.KeyedVectors.load(file_path) 
            assert type(to_return)==gensim.models.KeyedVectors
            break
        except: 
            print(f"error occurred while loading file {file_path}") 
            continue 
    return to_return 
# ====================================== end of objects and methods ====================================





# ====================================== user prompts =============================================== 


if __name__ == '__main__': 
    while(True): 
        print(long_line)
        print(f"Train new model ('train')? Or check existing model ('check')? Or exit ('exit')?") 
        ans = prompt_for_str(f"Enter here: ", options={'train','check','exit'}) 
        if ans=='train': 
            word_vector = Train_new_model() 
            Save_wv(word_vector=word_vector) 
            continue
        elif ans=='check': 
            word_vector = Prompt_for_wv() 
            Check_trained_model(word_vector=word_vector) 
            continue 
        elif ans=='exit': 
            break 
    exit(0) 