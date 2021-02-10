
import gensim
from gensim.models import Word2Vec

from Utilities import * 
import os 
import pickle 
from Clip import *
from Tokenizer_kit import *



# viewing pkl file
pkl_file = (open("chat_words/wardellwords.pkl", "rb")) 
data = pickle.load(pkl_file)

#print(data[:100])
model1 = gensim.models.Word2Vec(data, min_count=5, size=500, sg=1)
print("pog:amazing",model1.wv.similarity('pog', 'amazing'))
print("pog:pogchamp",model1.wv.similarity('pog', 'pogchamp'))
print("pog:sadge",model1.wv.similarity('pog', 'sadge'))
print("lol:lmao",model1.wv.similarity('lol', 'lmao'))
print("most similar to pog",model1.wv.most_similar('pog'))
print("most similar to lol",model1.wv.most_similar('lol'))
#print(model1.wv.get_vector('pog'))
