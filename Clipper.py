#!/usr/bin/python
# -*- coding: utf-8 -*-
from Utilities import * 
import numpy as np 
import os 
from Clip import * 
import json 
import pickle 
import copy 

''' This module prompts user to enter json file name for twitch stream chat
    The json file must be formatted as twitch chat format

    User will enter parameters, or choose to use default values'''
    
# ===================================== helper functions ============================================
    
# turn a twitch chat data loaded form json file into desired data structures
def Twitch_Comment_to_data(comments:list, chat_window=10, ignore_notices=True): 
    ''' takeis a twitch comment formatted dict, outputs tuple of four items
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


# make clips from passed chat (output of Twitch_comment_to_data) using chat speed
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
                clip.add_chat(time_points[i], chats[i])
                i+=1 
            if (i>=N and len(clip)>=min_len): 
                clip.end_time = time_points[-1] 
                clips.append(clip) 
            elif len(clip) >= min_len:
                clip.end_time = time_points[i]
                clips.append(clip)
            
        i+=1
        
    return clips 


# ========================================= main =======================================================
if __name__ == '__main__': 
    while(True):
        file_path = input('Enter json file path (WITH .json, enter exit to exit): ') 
        
        if type(file_path) != str: 
            print('invalid value entered, try again') 
            continue 
        elif file_path == 'exit': 
            exit(0) 
        elif not os.path.isfile(file_path): 
            print('file path entered invalid, try again') 
            continue 
        else: 
            break 
        
        
    chat_window = 10 
    min_clip_len = 10
    chat_speed_threshold = 100 
    ignore_notices = True 
    while(True): 
        answer = prompt_for_int('Do you want ot use default values? (1 for yes, -1 for no, 0 for more info): ', min_v=-1, max_v=1) 
        if answer == 0: 
            print(long_line)
            print(f'value for chat window is {chat_window}')
            print(f'the value for min_clip_len is {min_clip_len}')
            print(f'the value for threshold is {chat_speed_threshold}') 
            print(long_line)
            continue
        elif answer == 1: 
            break 
        elif answer == -1: 
            while(True): 
                chat_window_answer = prompt_for_int('Enter a chat window as an int (0 for more info): ', min_v=0)
                if chat_window_answer == 0: 
                    print(long_line)
                    print('chat window means how many chat messages we measure chat speed') 
                    print('for example, window os 1 is that we measure chat speed at every chat message') 
                    print('window of 10 means we average chat speed for every 10 chat: ')
                    print('so we have a speed for chat[0:10], and another for chat[1:11]')
                    print(f'the default value is {chat_window}')
                    print(long_line)
                    continue
                elif chat_window_answer > 0: 
                    chat_window = chat_window_answer
                    break 
                    
            while(True):  
                clip_len_answer = prompt_for_int('Enter minimum clip length as an int (0 for more info): ',min_v=0) 
                if clip_len_answer == 0: 
                    print(long_line)
                    print('clip length means how many chat minimum do we need in a clip') 
                    print('for example, if minimum is 10')
                    print('if a clip has 9 or less chat messages in it, we discard it') 
                    print('the default value is 10') 
                    print(long_line)
                    continue
                elif clip_len_answer > 0: 
                    min_clip_len = clip_len_answer  
                    break 
            
            while(True): 
                speed_thr_ans = prompt_for_int('Enter chat speed threshold in percent (0 for more info): ',min_v=0) 
                if speed_thr_ans == 0: 
                    print(long_line)
                    print('speed threshold means at what chat speed do we start clipping') 
                    print('each vod has its average chat speed, we take a percentage based on that') 
                    print('for example, if the answer is 120 and a vod has average chat speed of 5/second') 
                    print('then clipping will start when chat speed exceed 6/second') 
                    print('the default is 100 percent')
                    print(long_line)
                    continue 
                elif speed_thr_ans > 0: 
                    chat_speed_threshold = speed_thr_ans
                    break   
            
            break 
        
    while(True): 
        ans = input('Do you want to ignore notice chats? (y/n/i, i for more info): ') 
        if not(ans in {'y','n','i'}): 
            print(prompt_err_msg) 
            continue 
        elif ans == 'y': 
            break 
        elif ans == 'n': 
            ignore_notices = False 
            break 
        else: 
            print(long_line)
            print('Typically in twitch chat, every subscription is considered a chat') 
            print('If someone gift subs to someone, those will maximize chat speed in that interval')
            print('you can choose to ignore those or take those into account.') 
            print('Default value is True')
            print(long_line)
            continue
        
        
            
    try: 
        with open(file_path, encoding='utf-8') as f: 
            data = json.load(f) 
            
        comments = data['comments'] 
    except: 
        print('file format is not correct, exiting')
        exit(0)
    speed, chats, x, video_id = Twitch_Comment_to_data(comments=comments, chat_window=chat_window, ignore_notices=ignore_notices) 
    clips = Clip_from_Chat(speed, chats=chats, time_points=x, video_id=video_id, min_len=min_clip_len, threshold=chat_speed_threshold) 
                
    
    print(f'Number of clips found is {len(clips)}')
    while(True): 
        clip_file_name = input('How do you want to name this pickle file? (WITHOUT .pkl): ') 
        while(True):
            ans = input(f'Name will be \"{clip_file_name}.pkl\", are you sure? (y/n): ') 
            if not(ans in {'y', 'n'}): 
                print('invalid value entered, try again') 
                continue
            else: 
                break 
        if ans=='y': 
            break 
        elif ans=='n': 
            continue
            
        
    if (not os.path.exists('clip_data')) or (not os.path.isdir('clip_data')): 
        os.makedirs('clip_data')
    with open(os.path.join('clip_data', clip_file_name + '.pkl'), 'wb') as f: 
        pickle.dump(clips, f)
    exit(0)
    