from Tokenizer_kit import * 
import numpy as np 
from Embedding import *
from Utilities import *
from Data_loader import *
from collections import defaultdict 
from Clip import *
import gensim 
import random 
from gensim.models import KeyedVectors


'''
================================== Module Description =====================================================
This module is implemented to be a collection of methods for converting one data structure into another
For example: turn a corpus of sentences into one-hot vector mapping of each tokens
calling the main function in this module will prompt in terminal to test module methods on data 
Other modules can import this module to use its methods 
'''

'''========================================================= Supportive Methods =============================================================='''
# Turns a clip object into a tuple of (chat list, label int)
def Clip_to_tuple(clip:clip_it, binary=True) -> tuple: 
    if binary: 
        label = clip.get_label_binary()**2 
    else: 
        label = clip.get_label()
    return (clip.chats, label)


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


# compute magnitude of numpy vector 
def Magnitude_of(vector:np.ndarray): 
    assert len(vector.shape)==1, "passed a multi-dimention vector"
    return np.sqrt(vector.dot(vector)) 


# Normalize a vector to be magnitude of 1 
def Normalize_vector(vector:np.ndarray) -> np.ndarray: 
    ''' This function normalize the passed vector to have magnitude of 1''' 
    return vector/(Magnitude_of(vector)+np.finfo(dtype=np.float32).eps)


'''========================================================== One Hot Vector ================================================================='''
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

    
# Test One_hot_vectorizer on a list of clips 
def Test_ohv(clip_list:list, binary=True): 
    print(short_line) 
    if len(clip_list)==0: 
        print(f"You passed 0 clips") 
        return 
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
    
     
'''========================================================= Multi Layered Perceptron data converter ============================================'''

# Turn a word into numpy vector in keyed vector, 0 vector if not in vocab
def Word_to_vector(word:str, kv:KeyedVectors) -> np.ndarray: 
    ''' Turns a word into tensor vector, returns 0 if word not in vocab'''
    vector = Vector_of(word_vector=kv, word=word) 
    if type(vector)!=np.ndarray: 
        vector = np.zeros(kv.vector_size, dtype=np.float32) 
    vector = vector.astype(np.float32)
    return vector


# Turns chat from one clip to a single vector 
def Chat_to_1d_vec(chat_list:list, kv:KeyedVectors, threshold=0.75, topk=-1) -> np.ndarray: 
    ''' Takes a chat as a list of string, tokenize it with embedding tokenizer
        sort tokens based on decreasing frequency, 
        either take tokens until threshold of total chat is accounted for (default)
        or take the topk tokens 
        use their kv vectors, add and normalize them, return the result'''
    if topk<0: threshold=1.0 
    chat = Concatenate_str_list(chat_list, splitter=' ') 
    token_list = Embedding_tokenize(chat) 
    threshold *= len(token_list)
    token_freq = List_to_bow(token_list) 
    token_list = sorted(token_freq.keys(), key=token_freq.get, reverse=True) 
    vector = np.zeros(kv.vector_size, dtype=np.float32) 
    accounted=0
    for token in token_list: 
        vector = vector + Word_to_vector(token, kv) 
        topk-=1 
        accounted+=token_freq[token] 
        if topk==0 or accounted>threshold: 
            break 
    return Normalize_vector(vector) 


# Turns a clip list into a tuple of (chat 2d vector, label 2d vector)
def Clip_list_2_mlp_data(clip_list:list, kv:KeyedVectors, threshold=0.75, topk=-1, binary=True) -> tuple: 
    ''' Data from each clip is extracted, label depends on binary or not
        Turns the chat list into a single vector using kv 
        turns the label into a label vector 
        concate them into two 2d vectors''' 
    tup_list = Clip_list_to_tuples(clip_list,binary)
    chat_vecs = np.array([Chat_to_1d_vec(i[0],kv,threshold,topk) for i in tup_list],dtype=np.float32)
    label_vecs = np.array([i[1] for i in tup_list]) 
    v_size = np.max(label_vecs)+1 
    Y = np.zeros((label_vecs.shape[0],v_size), dtype=np.float32) 
    Y[:,label_vecs]=1.0
    return (chat_vecs, Y)
   
   
# Test Multi Layered Perceptron
def Test_mlp(clip_list:list, kv:KeyedVectors): 
    print(short_line) 
    if len(clip_list)==0: 
        print(f"You passed 0 clips") 
        return 
    ans=prompt_for_str(f"Do you want binary labels? (y/n): ", options={'y','n'}) 
    binary=(ans=='y')
    print(f"Testing mlp data loader on [{len(clip_list)}] clips")
    X,Y = Clip_list_2_mlp_data(clip_list,kv) 
    print(f"Each word turns into a vector of size: [{kv.vector_size}]")
    print(f"X is a vector of shape: {X.shape}") 
    print(f"Y is a vector of shape: {Y.shape}") 
    return 
    
    
'''================================================= RNN data loader ================================================''' 












'''========================================================= main function ==================================================================='''
        
def main(): 
    while(True): 
        print(long_line)
        print(f"Enter testing options: ") 
        print(f"[ohv]: to test one-hot vector") 
        print(f"[mlp]: to test multi-layered perceptron data loader")
        print(f"[rnn]: to test rnn data loader")
        print(f"[e]: exit")
        ans=prompt_for_str("Enter here: ", options={'ohv','mlp','rnn','e'}) 
        if ans=='e': break 
        print(f"Enter your testing data")
        clip_list = Prompt_for_data() 
        if ans=='ohv': 
            Test_ohv(clip_list) 
            continue 
        elif ans=='mlp': 
            Test_mlp(clip_list, Load_wv()) 
            continue 
        elif ans=='rnn': 
            continue
    return 



if __name__=='__main__': 
    main() 
    exit(0)