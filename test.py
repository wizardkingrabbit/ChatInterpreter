#!/usr/bin/python
# -*- coding: utf-8 -*-

from Utilities import * 
from Clip import * 
from Tokenizer_kit import * 
import pickle 
from Data_loader import * 
import torch
import os
import random
import time
import math
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np 
from Embedding import * 
import gensim 


if __name__ == '__main__': 
    # prompt user to input pkl

    with open('labeled_clip_data/Teo/TeosGame[0]_labeled.pkl', 'rb') as f:
        data = pickle.load(f)

        for i in data:
            print(i.label)