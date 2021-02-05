# this module defines tokenizing tools for us to use in leaner modules. 
# typically tokanizing take a long string
# thus any processing to get the long string should be done outside the tokenizing functions

import os 
import numpy as np 

# ================================= this part is for methods to make long string from collections =======================


def concatenate_str_list(str_list:list, random_order:False, splitter=os.linesep) -> str: 
    ''' This function take a list of string, concatenate them into a long string 
        random order means items are concatenated in random order 
        splitter is what to put in between items in the long string''' 
        
    if not random_order: 
        to_return = splitter.join(str_list) 
        return to_return
    else: 
        index_arr = np.arange(len(str_list), dtype=int)
        index_arr = np.random.shuffle(index_arr) 
        to_return = '' 
        for i in index_arr: 
            assert type(str_list[i]) == str, 'concatenating non string items' 
            to_return += str_list[i] 
            to_return += splitter  
        return to_return 
    
    

# =============================== this part is for processing long strings ==============================================