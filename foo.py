from Utilities import * 
import os 
import pickle 


with open('clip_data/TeosGame.pkl', 'rb') as f: 
    data = pickle.load(f) 
    
    
print(type(data)) 
print(len(data)) 
print(data[0])
data = data[0].copy() 
print(data.start_time_to_str()) 
print(data.end_time_to_str()) 
print(data.chat_duration_to_str()) 
print(data.clip_duration_to_str())
print(data.label_info_to_str())