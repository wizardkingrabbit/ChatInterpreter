import numpy as np 
import os 
import json 
import pickle 
import copy 

long_line = '================================================================='
short_line = '---------------------------------------------------------------'
prompt_err_msg = 'invalid value entered, try again'

def Twitch_Comment_to_data(comments:dict, chat_window=10, ignore_notices=True): 
    ''' takes a twitch comment formatted dict, outputs tuple of four items
        first is speed, second is chat string data, third is time points of those chats, fourth is video id
        ignore_notices is a bool value that decide whether to ignore sub/resub notices'''
    video_id = comments[0]['content_id']
    offsets = list()
    chats = list() 
    ignored = 0
    for i in comments: 
        notice = i['message']['user_notice_params']['msg-id']
        if ignore_notices and not(notice in {'', None}): 
            ignored += 1
            continue
        offsets.append(i['content_offset_seconds']) 
        chats.append(i['message']['body']) 
        
    chats = chats[chat_window::]
    chats = np.char.array(chats, unicode=True) 
    offsets = np.array(offsets, dtype=float)
    speed = offsets[chat_window::] - offsets[:-chat_window:]
    x = np.array(offsets[chat_window::],dtype=float) 
    speed = speed + 1.0/chat_window
    speed = 1/speed 
    
    if ignore_notices: 
        print(f'Number of chats ignored is {ignored}')
    return (speed,chats,x,video_id)

def prompt_for_int(message:str, min_v=None, max_v=None) -> int: 
    ''' prompt for integer input with passed message and do all error checking,
        also check for min and max value range, they are inclusive'''
    while(True): 
        ans = input(message) 
        try: 
            ans = int(ans) 
            if min_v!=None: 
                assert ans>=min_v
            if max_v!=None: 
                assert ans<=max_v
            break
        except: 
            print('invalid value entered, try again') 
            continue 
        
    return ans 


def prompt_for_float(message:str, min_v=None, max_v=None) -> float: 
    ''' prompt for float input with passed message and do all error checking,
        also check for min and max value range, they are inclusive'''
    while(True): 
        ans = input(message) 
        try: 
            ans = float(ans) 
            if min_v!=None: 
                assert ans>=min_v
            if max_v!=None: 
                assert ans<=max_v
            break
        except: 
            print('invalid value entered, try again') 
            continue 
        
    return ans 




def time_to_str(time:float) -> str: 
    ''' return a formated str of time in h:m:s'''
    time = int(time) 
    hours = time // 3600 
    minutes = (time % 3600) // 60 
    seconds = time % 60 
    
    return (f'{hours}:{minutes}:{seconds}') 




# ==================================================== clip class =============================================================
class clip_it(): 
    
    available_labels = {0:'unlabeled',
                        1:'special topic',
                        2:'amasement',
                        3:'amusement',
                        4:'disappointment', 
                        5:'shock', 
                        6:'pure confusion', 
                        7:'other'}
    
    positive_labels = {2,3}
    negative_labels = {4,5,6}
    neutral_labels =  {0,1,7}
    
    # IMPORTANT: when you add a class, make sure to add one the same way as indexing and put that number in binary class as well
    
    def __init__(self, start:float, video_id:str, span_duration=5.0): 
        self.start_time = start
        self.chats = list() 
        self.end_time = -1
        self.label = 0
        self.span_duration = span_duration
        self.video_id = video_id

    def __len__(self): 
        return len(self.chats) 
    
    def __str__(self) -> str: 
        to_return = ''
        for i,c in enumerate(self.chats): 
            to_return += f'[{i}]: {c}' 
            to_return += os.linesep 
            
        return to_return
    
    def copy(self): 
        ''' make a copy of itself''' 
        to_return = clip_it(start=float(self.start_time), video_id=str(self.video_id), span_duration=float(self.span_duration)) 
        to_return.end_time = float(self.end_time)
        to_return.label = int(self.label)
        to_return.chats = copy.deepcopy(self.chats) 
        
        return to_return
        
    
    def is_valid(self) -> bool: 
        ''' check if the clip object is valid'''
        return (self.end_time>0.0) and (self.start_time>0.0) and (self.end_time>self.start_time) and (len(self.chats)>0) 
    
    def chat_duration(self) -> float: 
        ''' returns the duration is the clip in floats unit is seconds ''' 
        assert self.is_valid() 
        return self.end_time - self.start_time 
    
    def set_label(self, n:int): 
        ''' sets the label for this clip, entered value must be within label index'''
        
        assert n < len(self.available_labels), 'entered invalid value in clip labeling' 
        
        self.label = n
        
        
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
        
        
        
        
        
        
# ======================================================== end of class ===============================================================
        

    
      



def Clip_from_Chat(speed, chats, time_points, video_id, min_len=5, threshold = 100): 
    ''' takes chat speed and string numpy array, turn into a list of clip class objects 
        it will not clip if clip chat is less then min_len in length, or below threshold% of average speed'''
    assert len(speed) == len(chats)
    assert len(chats) == len(time_points)
    N = len(speed) 
    i = 0
    avg_speed = speed.mean()
    speed_threshold = (float(threshold)/100.0) * float(avg_speed)
    clips = list()
    while(i<N):
        if speed[i] >= avg_speed: 
            clip = clip_it(time_points[i], video_id)
            while(i<N and speed[i] >= speed_threshold): 
                clip.chats.append(chats[i]) 
                i+=1 
            if len(clip) >= min_len:
                clip.end_time = time_points[i]
                clips.append(clip)
            
        i+=1
        
        
    return clips 
