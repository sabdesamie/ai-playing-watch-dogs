# test_model.py
import tensorflow as tf
import numpy as np
import cv2
from PIL import ImageGrab
import time
from get_keys import key_check
import os
from grabscreen import grab_screen
from alexnet import alexnet
from directkey import PressKey, ReleaseKey,A,W,D


WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 10



t_time = 0.09

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

model = alexnet(WIDTH,HEIGHT,LR)
#model.load("model_alexnet-154000")
model.load("wd_model_v3.h5")


last_time = time.time()
paused = False


while(True):

    if not paused:

        #screen = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
        screen = grab_screen(region = (0,40,800,640))
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(80,60))

        # print(time.time()-last_time)
        last_time = time.time()

        prediction = model.predict([screen.reshape(80,60,1)])[0]
        moves = list(np.around(prediction))
        print(prediction,moves)


        if moves == [1,0,0]:
            left()
        elif moves == [0,1,0]:
            straight()
        elif moves == [0,0,1]:
            right()

    keys = key_check()

    if 'T' in keys:
        if paused:
            paused= True
            time.sleep(1)
        else:
            paused=True
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            time.sleep(1)




