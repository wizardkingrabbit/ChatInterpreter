from Utilities import * 
import numpy as np 
import os 
import json 
import pickle 
import copy
import Clip
if __name__ == "__main__":
    file_name = prompt_for_file("give a labeled pickle file -> ")
    count_a = 0
    count_b = 0
    with open(file_name, "rb") as f:
        clip_list = pickle.load(f)
    for clip in clip_list:
        count_a += 1
        if (clip.get_label_binary() != 0):
            count_b += 1
            print(clip.start_time_to_str())
    print(count_a)
    print(count_b)
    while (input("e -> ") != "e"):
        pass
