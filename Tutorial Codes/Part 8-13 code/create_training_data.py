# create_training_data.py

import numpy as np
import cv2
from PIL import ImageGrab
import time
from get_keys import key_check
import os
from grabscreen import grab_screen

def keys_to_output(keys):
    output = [0,0,0]
    if "Q" in keys:
        output[0] = 1
    elif "D" in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output

for i in list(range(4)):
    print(4-i)
    time.sleep(1)

file_name = "watch_dogs_119_training_data.npy"

if os.path.isfile(file_name):
    print("file exist")
    training_data = list(np.load(file_name,allow_pickle=True))
else:
    print("file does not exist")
    training_data = []



while(True):
    screen = grab_screen(region = (0,40,800,640))
    screen = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen,(80,60))
    keys = key_check()
    output = keys_to_output(keys)
    training_data.append([screen,output])


    if len(training_data) % 1000 == 0:
        print(len(training_data))
        np.save(file_name,training_data)
