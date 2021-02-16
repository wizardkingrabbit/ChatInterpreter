#!/usr/bin/python
# -*- coding: utf-8 -*-
# clip it class



from Utilities import * 
import copy 

class clip_it(): 
    
    available_labels = {0:'unlabeled',
                        1:'special topic',
                        2:'amasement',
                        3:'amusement',
                        4:'disappointment', 
                        5:'shock, scared', 
                        6:'pure confusion', 
                        7:'other', 
                        8:'unfortunate (nice try)'} 
    
    positive_labels = {2,3,8}
    negative_labels = {4,5,6}
    neutral_labels =  {0,1,7}
    
    # IMPORTANT: when you add a class, make sure to add one the same way as indexing and put that number in binary class as well
    
    def __init__(self, start:float, video_id:str, span_duration=5.0): 
        self.start_time = start
        self.chats = list() 
        self.time_stamps = list()
        self.end_time = -1
        self.label = 0
        self.span_duration = span_duration
        self.video_id = video_id 
        self.pred_label = 0 
        self.pred_binary_label = 0 
        

    def __len__(self): 
        return len(self.chats) 
    
    def __str__(self) -> str: 
        to_return = ''
        for i in range(len(self)): 
            to_return += f'[{i}] [{time_to_str(self.time_stamps[i])}]: {self.chats[i]}' 
            to_return += os.linesep 
            
        to_return += (f"true label is [{self.label}]: {self.available_labels[self.label]}" + os.linesep)
        to_return += (f"true binary label is [{self.get_label_binary()}]" + os.linesep) 
        to_return += (f"predicted label is [{self.pred_label}]: {self.available_labels[self.pred_label]}" + os.linesep) 
        to_return += (f"predicted binary label is [{self.pred_binary_label}]" + os.linesep)
            
        return to_return
    
    def add_chat(self, T:float, chat:str): 
        assert T>=0.0 
        self.time_stamps.append(float(T)) 
        self.chats.append(str(chat)) 
        return None 
    
    
    def copy(self): 
        ''' make a copy of itself''' 
        to_return = clip_it(start=float(self.start_time), video_id=str(self.video_id), span_duration=float(self.span_duration)) 
        to_return.end_time = float(self.end_time)
        to_return.label = int(self.label)
        to_return.chats = copy.deepcopy(self.chats) 
        to_return.time_stamps = copy.deepcopy(self.time_stamps)
        
        return to_return
        
    
    def is_valid(self) -> bool: 
        ''' check if the clip object is valid'''
        if (self.start_time<0.0) or (self.end_time<self.start_time): 
            return False 
        elif (len(self.chats)<1) or (len(self.chats)!=len(self.time_stamps)): 
            return False
        else: 
            return True
    
    
    def chat_duration(self) -> float: 
        ''' returns the duration is the clip in floats unit is seconds ''' 
        assert self.is_valid() 
        return self.end_time - self.start_time 
    
    
    def set_label(self, n:int): 
        ''' sets the label for this clip, entered value must be within label index'''
        
        assert n in self.available_labels.keys(), 'entered invalid value in clip labeling' 
        
        self.label = n
        
        
    def set_pred_label(self, n:int): # set the prediction label of the clip 
        assert n in self.available_labels.keys(), f"Entered label not valid" 
        self.pred_label = n 
        
        
    def set_pred_binary_label(self, n:int): # set the prediction of binary label 
        assert n in {-1,1}, f"Entered invalid label" 
        self.pred_binary_label=n 
    
    
    def set_span_duration(self, span:float): 
        self.span_duration = span
        
        
    def get_label(self) -> str: 
        ''' returns the label string for this clip''' 
        return self.available_labels[self.label] 
    
    
    def get_label_binary(self) -> int: 
        ''' returns clip label in binary classification 
            1 is positive, -1 is negative, 0 is unlabeled (or something is wrong)
            typically, we do not want to use unlabeled, but it is up to the model maker''' 
        if self.label in self.neutral_labels: 
            return 0 
        elif self.label in self.positive_labels: 
            return 1 
        elif self.label in self.negative_labels: 
            return -1
        else: 
            return 0 
        
            
    def get_pred_label(self) -> int: # return the prediction label 
        return self.pred_label 
    
    
    def get_pred_label_binary(self) -> int: # return the predicted binary label 
        return self.pred_binary_label
         
    
    def label_info(self) -> dict: 
        ''' return the possible label dict for the clips''' 
        return self.available_labels 
    
    
    def label_info_to_str(self) -> str: 
        ''' return lable info as a printable str'''
        to_return = ''
        to_return += os.linesep
        for i,x in self.available_labels.items(): 
            to_return += f'label index {i} is [{x}]'
            to_return += os.linesep
            
        to_return += f'to access this dict, use method .label_info(){os.linesep}'
        to_return += short_line
        
        return to_return 
    
    

        
    def start_time_to_str(self) -> str: 
        ''' return start time in formated string'''
        return time_to_str(self.start_time) 
    
    def end_time_to_str(self) -> str: 
        ''' return end time in formated string'''
        return time_to_str(self.end_time)

        
        
    def chat_duration_to_str(self) -> str: 
        ''' return chat duration in string format'''
        to_return = 'from ' 
        to_return += self.start_time_to_str() 
        to_return += ' to ' 
        to_return += self.end_time_to_str() 
        
        return to_return 
    
    
    def clip_duration_to_str(self) -> str: 
        ''' return formated string of clip duration, 
            it is span duration times chat duration before chat start time''' 
            
        to_return = 'from ' 
        span = float(self.end_time - self.start_time) 
        span *= self.span_duration 
        clip_start_at = float(self.start_time - span) 
        if clip_start_at<0: 
            clip_start_at = 0.0
        clip_start_time = time_to_str(clip_start_at)
        
        to_return += clip_start_time 
        to_return += ' to '
        to_return += self.end_time_to_str() 
        
        return to_return
        
        
        