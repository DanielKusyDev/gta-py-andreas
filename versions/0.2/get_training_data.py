import os
import time

import cv2
import numpy as np

import services

gatherer = services.TrainingDataGatherer(800, 600, 0, 0)
starting_file = int(input("Starting file number: "))
for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)

file_name = 'data/training_data-{}.npy'
training_data = []

paused = False
new_file_threshold = 20
counter = 0
while True:
    if not paused:
        screen, output = gatherer.gather_data(160, 120)
        training_data.append([screen, output])
        if len(training_data) % 1000 == 0:
            print(f"Saving {len(training_data)} samples")
            np.save(file_name.format(starting_file), training_data)
            counter += 1
            if new_file_threshold == counter:
                training_data = []
                counter = 0
                starting_file += 1
                print(f"Starting new file - {file_name.format(starting_file)}")
    keys = services.get_pressed_keys()
    if 'T' in keys:
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            time.sleep(1)
    k = cv2.waitKeyEx(25)
    if k == 27:
        break