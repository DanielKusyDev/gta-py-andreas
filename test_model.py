import time

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import cv2

from services import AbstractDriver, TrainingDataGatherer, get_pressed_keys


def create_model(width, height, output=3, LR=1e-3):
    model = models.Sequential()

    # model.add(layers.InputLayer())
    # 1st conv
    model.add(layers.Conv2D(input_shape=[width, height, 1], filters=96, kernel_size=11, strides=(4, 4),
                            padding="valid", activation="relu"))
    # 1st pooling
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.BatchNormalization())

    # 2nd conv
    model.add(layers.Conv2D(filters=256, kernel_size=5, activation="relu",
                            padding="same", kernel_initializer='he_normal'))
    # 2nd pooling
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.BatchNormalization())

    # 3rd conv
    model.add(layers.Conv2D(filters=384, kernel_size=3, activation="relu",
                            padding="same", kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())

    # 4th conv
    model.add(layers.Conv2D(filters=384, kernel_size=3, activation="relu",
                            padding="same", kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())

    # 5th conv
    model.add(layers.Conv2D(filters=256, kernel_size=3, activation="relu",
                            padding="same", kernel_initializer='he_normal'))

    # 3rd pooling
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.BatchNormalization())

    # passing to dense layer
    model.add(layers.Flatten())

    # 1st dense
    model.add(layers.Dense(4096, input_shape=(224 * 224 * 3,), activation="relu"))

    model.add(layers.Dropout(rate=0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=output, activation="softmax"))
    model.compile(optimizer=optimizers.Adam(LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


class GtaDriver(AbstractDriver):

    def drive(self, prediction, fwd_threshold, left_threshold, right_threshold, full_speed):
        result = "Going: {direction} \t->\t {pred}"
        # if prediction[1] > fwd_threshold:
        #     if full_speed:
        #         self.go_straight()
        #     result = result.format(direction="Forward", pred=str(np.around(prediction[1], 5)))
        # el
        if prediction[0] > left_threshold:
            self.go_left(full_speed)
            result = result.format(direction="Left", pred=str(np.around(prediction[0], 4)))
        elif prediction[2] > right_threshold:
            self.go_right(full_speed)
            result = result.format(direction="Right", pred=str(np.around(prediction[2], 4)))
        else:
            self.go_straight(full_speed)
            result = result.format(direction="Forward", pred=str(np.around(prediction[1], 5)))
        return result


WIDTH = 160
HEIGHT = 100
LR = 1e-3
EPOCHS = 10
samples = "200k"
MODEL_NAME = f"models/alexnet-noaug-flip-{samples}/checkpoint"

model = create_model(WIDTH, HEIGHT, 3, LR)
model.load_weights(MODEL_NAME)
#
for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)

driver = GtaDriver()
width = 800
height = 600
gatherer = TrainingDataGatherer(width, height)
counter = 0
speed_thresh = 5
stop = 3
last_time = time.time()
paused = False

while True:
    try:
        if not paused:
            counter += 1
            screen = gatherer.grab_screen()
            screen = cv2.resize(screen, (160, 100))
            pts = np.array([[0, 70], [0, 120], [160, 120], [160, 0], [160, 70], [120, 60], [55, 60]])
            croped = screen
            pts = pts - pts.min(axis=0)
            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(croped, croped, mask=mask)
            screen = dst
            # cv2.imshow("Recorder", dst)
            screen = screen.reshape(-1, 160, 100, 1)
            prediction = model.predict(screen)
            prediction = prediction[0]
            prediction = np.multiply(prediction, [0.1, 1, 1000])
            print(round(prediction[0], 4), round(prediction[1], 4), round(prediction[2], 4))
            # 0.19 0.1 0.1
            res = driver.drive(prediction, left_threshold=0.91, fwd_threshold=0.1,
                               right_threshold=0.1, full_speed=counter < speed_thresh)
            # print(res)
            if counter >= speed_thresh * 2:
                counter = 0


        keys = get_pressed_keys()
        k = cv2.waitKeyEx(25)
        if k == 27:
            break

        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                driver.stop()
                time.sleep(1)
    except Exception as e:
        print(str(e))
