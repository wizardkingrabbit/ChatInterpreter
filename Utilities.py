import numpy as np 
import os 
import json 
import pickle 

def Twitch_Comment_to_data(comments:dict, chat_window=10): 
    ''' takes a twitch comment formatted dict, outputs tuple of three numpy arrays
        first is speed, second is chat string data, third is time points of those chats'''
    offsets = list()
    chats = list() 
    for i in comments: 
        offsets.append(i['content_offset_seconds']) 
        chats.append(i['message']['body']) 
        
    chats = chats[chat_window::]
    chats = np.char.array(chats, unicode=True) 
    offsets = np.array(offsets, dtype=float)
    speed = offsets[chat_window::] - offsets[:-chat_window:]
    x = np.array(offsets[chat_window::],dtype=float) 
    speed = speed + 1.0/chat_window
    speed = 1/speed
    return (speed,chats,x)


class clip_it(): 
    def __init__(self, start:float):
        self.start_time = start
        self.chats = list() 
        self.end_time = -1

    def __len__(self): 
        return len(self.chats)  



def Clip_from_Chat(speed, chats, time_points, min_len=5, threshold = 100): 
    ''' takes chat speed and string numpy array, turn into a list of clip class objects 
        it will not clip if clip chat is less then 5 in length'''
    assert len(speed) == len(chats)
    assert len(chats) == len(time_points)
    N = len(speed) 
    i = 0
    avg_speed = speed.mean()
    speed_threshold = (float(threshold)/100.0) * float(avg_speed)
    clips = list()
    while(i<N):
        if speed[i] >= avg_speed: 
            clip = clip_it(time_points[i])
            while(i<N and speed[i] >= speed_threshold): 
                clip.chats.append(chats[i]) 
                i+=1 
            if len(clip) >= min_len:
                clip.end_time = time_points[i]
                clips.append(clip)
            
        i+=1
        
        
    return clips 
