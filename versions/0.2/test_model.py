import time

import cv2
from PIL import ImageGrab
import numpy as np

from services import AbstractDriver, get_pressed_keys, TrainingDataGatherer

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir='log')

    return model
#
#
class GtaDriver(AbstractDriver):

    def drive(self, prediction, fwd_threshold, left_threshold, right_threshold):
        result = "Going: {direction} \t->\t {pred}"

        if prediction[1] > fwd_threshold:
            self.go_straight()
            result = result.format(direction="Forward", pred=str(np.around(prediction[1], 5)))
        elif prediction[0] > left_threshold:
            self.go_left()
            result = result.format(direction="Left", pred=str(np.around(prediction[0], 5)))
        elif prediction[2] > right_threshold:
            self.go_right()
            result = result.format(direction="Right", pred=str(np.around(prediction[2], 5)))
        else:
            self.go_straight()
            result = result.format(direction="Forward", pred=str(np.around(prediction[0], 5)))
        return result


WIDTH = 160
HEIGHT = 100
LR = 1e-3
EPOCHS = 10
CNN = "alexnet"
MODEL_NAME = "models/alexnet-samples-100k-epochs-10-lr-0.001.models"
model = alexnet(160, 60, LR)
model.load(MODEL_NAME)
for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)

driver = GtaDriver()
width = 800
height = 600
gatherer = TrainingDataGatherer(width, height)
counter = 0
last_time = time.time()
paused = False
while True:
    try:
        if not paused:
            counter += 1
            screen = gatherer.grab_screen()
            screen = cv2.resize(screen, (160, 120))
            pts = np.array([[0, 70], [0, 120], [160, 120], [160, 0], [160, 70], [120, 60], [55, 60]])
            croped = screen
            pts = pts - pts.min(axis=0)
            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(croped, croped, mask=mask)
            screen = dst[60:120, 0:160]
            cv2.imshow("recorder", screen)
            screen = screen.reshape(-1, 160, 60, 1)
            prediction = model.predict(screen)[0]
            prediction = np.multiply(prediction, [0.8, 1, 0.705])
            #
            res = driver.drive(prediction, fwd_threshold=0.4, left_threshold=0.7,
                               right_threshold=0.702)
            print(f"{res}\tTook: {time.time()}")

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
                # driver.stop()
                time.sleep(1)
    except Exception as e:
        print(str(e))
