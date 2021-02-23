from Data_loader import * 

# data = Load_clips_from_file('mislabeled/teo.pkl') 
with open('mislabeled/teo.pkl','rb') as f: 
    data = pickle.load(f) 
print(len(data))  
print(type(data)) 
print(type(data[0]))