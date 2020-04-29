# train_model.py

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
EPOCHS = 4
MODEL_NAME = "pywd_model_v3_final.h5"

model = alexnet(WIDTH,HEIGHT,LR)
model.load(MODEL_NAME)


train_data = np.load('watch_dogs_total_balanced_data.npy',allow_pickle=True)
train = train_data[:-100]
test = train_data[-100:]


for i in range(EPOCHS):


        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save("wd_model_v3.h5")


# tensorboard --logdir=foo:C:/path/to/log





