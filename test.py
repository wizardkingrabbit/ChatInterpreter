
import gensim
from gensim.models import Word2Vec

from Utilities import * 
import os 
import pickle 
from Clip import *
from Tokenizer_kit import *



# viewing pkl file
pkl_file = (open("chat_words/TeosGame.pkl", "rb")) 
data = pickle.load(pkl_file)

custom_stop_words = set(nltk_stop_words) 
custom_stop_words.update({',', '\'', '(', ')', '.', '@'})


processed_data = list() 
i=0
while (i < len(data)): 
    i+=100 
    temp = data[i-100:i] 
    temp = Concatenate_str_list(str_list=temp) 
    to_append = Simple_tokenizer(long_string=temp, stop_words=custom_stop_words)
    processed_data.append(to_append) 




def Compare_two_words(w1:str, w2:str, w_vector): 
    print(long_line)
    print(f"{w1}:{w2} has similarity {w_vector.similarity(w1, w2)}") 
    print(f">> words similar to {w1} are: ") 
    for w,v in w_vector.most_similar(w1): 
        print(f">>>> {w}: {v}") 
    
    print(short_line)
    print(f">> words similar to {w2} are:") 
    for w,v in w_vector.most_similar(w2): 
        print(f">>>> {w}: {v}") 
        
    print(long_line)
        

    
# print(processed_data[100:120])
#print(data[:100])
model1 = gensim.models.Word2Vec(processed_data, min_count=20, size=300, window=7, iter=5)
word_vectors = model1.wv 
# print("pog:amazing",model1.wv.similarity('pog', 'amazing'))
# print("pog:pogchamp",model1.wv.similarity('pog', 'pogchamp'))
# print("pog:sadge",model1.wv.similarity('pog', 'sadge'))
# print("lol:lmao",model1.wv.similarity('lol', 'lmao'))
# print("lmao:sadge",model1.wv.similarity('lmao', 'sadge'))
# # print("most similar to pog",model1.wv.most_similar('pog'))
# # print("most similar to lol",model1.wv.most_similar('lol')) 
# print(f"kekw:lol similarity is {vectors.similarity('kekw', 'lol')}")
# print(f"kekw:lul similarity is {vectors.similarity('kekw', 'lul')}")
# print(f"most similar to kekw {vectors.most_similar('kekw')}")


#Compare_two_words('pog', 'pogg', word_vectors) 
#Compare_two_words('pogg','poggggg', word_vectors)
Compare_two_words('monkas', 'lul', word_vectors)
Compare_two_words('nice', 'noice', word_vectors) 
Compare_two_words('pog', 'kekw', word_vectors) 
Compare_two_words('kekw', 'sadge', word_vectors) 
Compare_two_words('kekw', 'pepelaugh', word_vectors) 
Compare_two_words('sadge', 'd', word_vectors) 
Compare_two_words(':', 'd', word_vectors)

#print(model1.wv.get_vector('pog'))
