# balance_data.py

import numpy as np
from random import shuffle

train_data = []

for i in range(105):
    
    # Because of correpted files .
    if (i != 5 and i != 6 and i != 8 and i!= 14 and i != 20 and i != 26 and i != 101):
        train_data += list(np.load("watch_dogs_"+str(i)+"_training_data.npy",allow_pickle=True))

forwards = []
lefts = []
rights = []

for data in train_data:

    screen = data[0]
    move = data[1]

    if move == [1,0,0]:
        lefts.append([screen,move])
    elif move == [0,1,0]:
        forwards.append([screen,move])
    elif move == [0,0,1]:
        rights.append([screen,move])

SIZE = min(len(lefts),len(rights),len(forwards))

lefts = lefts[:SIZE]
rights = rights[:SIZE]
forwards = forwards[:SIZE]

train_data = lefts + rights + forwards

shuffle(train_data)
print(len(train_data))

np.save('watch_dogs_total_balanced_data.npy', train_data)



