# this is the template learner, do not change this file but make copies and name them accordingly


# import all you need
import os 
from Utilities import * 
from Clip import * 
from Tokenizer_kit import * 



# ======================================= objects and methods =======================================









# ====================================== end of objects and methods ====================================





# ====================================== user prompts =============================================== 


if __name__ == '__main__': 
    
    clip_path_list = [] 
    
    while(True):
        file_path = input('Enter pkl clip file path (WITH .pkl, enter exit to exit, done to proceed): ') 
        
        if type(file_path) != str: 
            print('invalid value entered, try again') 
            continue 
        elif file_path == 'exit': 
            exit(0) 
        elif file_path == 'done': 
            assert len(clip_path_list) > 0, 'no file entered'
            break 
        elif not os.path.isfile(file_path): 
            print('file path entered invalid, try again') 
            continue 
        else: 
            clip_path_list.append(file_path) 
    
    # now you have a list of clip file path, process them into data structure of your desire
    # then train and test the clips. 
    # make sure you print out enough result for human interpretation 
    
    
    
    exit(0) 