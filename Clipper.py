from Utilities import * 
import os 

''' This module prompts user to enter json file name for twitch stream chat
    The json file must be formatted as twitch chat format
    User will enter parameters, or choose to use default values'''

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
    while(True): 
        answer = input('Do you want ot use default values? (1 for yes, -1 for no, 0 for more info): ') 
        try: 
            answer = int(answer) 
        except: 
            print('invalid value entered, try again')
            continue 
        if answer not in {0,1,-1}: 
            print('invalid value entered, try again')
            continue 
        elif answer == 0: 
            print('============================================================')
            print(f'value for chat window is {chat_window}')
            print(f'the value for min_clip_len is {min_clip_len}')
            print(f'the value for threshold is {chat_speed_threshold}')
            print('============================================================')
            continue
        elif answer == 1: 
            break 
        elif answer == -1:
            while(True): 
                chat_window_answer = input('Enter a chat window as an int (0 for more info): ') 
                try: 
                    chat_window_answer = int(chat_window_answer) 
                except: 
                    print('invalid value entered, try again')
                    continue 
                if type(chat_window_answer) != int or chat_window_answer < 0: 
                    print('invalid value entered, try again') 
                    continue 
                elif chat_window_answer == 0: 
                    print('============================================================')
                    print('chat window means how many chat messages we measure chat speed') 
                    print('for example, window os 1 is that we measure chat speed at every chat message') 
                    print('window of 10 means we average chat speed for every 10 chat: ')
                    print('so we have a speed for chat[0:10], and another for chat[1:11]')
                    print(f'the default value is {chat_window}')
                    print('============================================================')
                    continue
                elif chat_window_answer > 0: 
                    chat_window = chat_window_answer
                    break 
                
            while(True): 
                clip_len_answer = input('Enter minimum clip length as an int (0 for more info): ') 
                try: 
                    clip_len_answer = int(clip_len_answer) 
                except: 
                    print('invalid value entered, try again')
                    continue 
                if clip_len_answer < 0: 
                    print('invalid value, try again') 
                    continue 
                elif clip_len_answer == 0: 
                    print('============================================================')
                    print('clip length means how many chat minimum do we need in a clip') 
                    print('for example, if minimum is 10')
                    print('if a clip has 9 or less chat messages in it, we discard it') 
                    print('the default value is 10') 
                    print('============================================================')
                    continue
                elif clip_len_answer > 0: 
                    min_clip_len = clip_len_answer 
                    break 
                    
            while(True): 
                speed_thr_ans = input('Enter chat speed threshold in percent (0 for more info): ') 
                try: 
                    speed_thr_ans = int(speed_thr_ans) 
                except: 
                    print('invalid value entered, try again')
                    continue 
                if speed_thr_ans < 0: 
                    print('value invalid, try again') 
                    continue 
                elif speed_thr_ans == 0: 
                    print('============================================================')
                    print('speed threshold means at what chat speed do we start clipping') 
                    print('each vod has its average chat speed, we take a percentage based on that') 
                    print('for example, if the answer is 120 and a vod has average chat speed of 5/second') 
                    print('then clipping will start when chat speed exceed 6/second') 
                    print('the default is 100 percent')
                    print('============================================================')
                    continue 
                elif speed_thr_ans > 0: 
                    chat_speed_threshold = speed_thr_ans 
                    break 
            break 
        
            
    try: 
        with open(file_path, encoding='utf-8') as f: 
            data = json.load(f) 
            
        comments = data['comments'] 
        speed, chats, x = Twitch_Comment_to_data(comments, chat_window) 
        clips = Clip_from_Chat(speed, chats, x, min_clip_len, chat_speed_threshold) 
    except: 
        print('file format is not correct, exiting')
        exit(0)
                
                
    while(True): 
        clip_file_name = input('How do you want to name this pickle file? (WITHOUT .pkl): ') 
        if type(clip_file_name) != str: 
            print('value invalid, try again') 
            continue 
        else: 
            break 
        
    if (not os.path.exists('clip_data')) or (not os.path.isdir('clip_data')): 
        os.makedirs('clip_data')
    with open(os.path.join('clip_data', clip_file_name + '.pkl'), 'wb') as f: 
        pickle.dump(clips, f)
    exit(0)
    