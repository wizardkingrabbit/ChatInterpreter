from Utilities import * 
import os 
import pickle 
from Clip import *


with open('clip_data/test.pkl', 'rb') as f: 
    data = pickle.load(f) 
    
    
sample = data[0] 
print(len(sample.time_stamps)) 
print(len(sample.chats)) 
