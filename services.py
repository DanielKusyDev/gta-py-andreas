import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import abc

from directkeys import ReleaseKey, PressKey, A, D, W

key_list = ["\b", *[char for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\"]]



def get_pressed_keys():
    keys = []
    for key in key_list:
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


class TrainingDataGatherer(object):

    def __init__(self, width, height, x_offset=0, y_offset=40):
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.img = None

    def keys_to_output(self, keys):
        '''
        Convert keys to a ...multi-hot... array
        [A,W,D] boolean values.
        '''
        output = [0, 0, 0]

        if 'A' in keys:
            output[0] = 1
        elif 'D' in keys:
            output[2] = 1
        else:
            output[1] = 1

        return output

    def grab_screen(self):
        width = self.width
        height = self.height
        # if self.x_offset <= 0:
        #     width = self.width - self.x_offset + 1
        #     height = self.height - self.y_offset + 1
        # else:


        hwin = win32gui.GetDesktopWindow()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (self.x_offset, self.y_offset), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.img

    def process_img(self, width, height):
        self.img = cv2.resize(self.img, (width, height))
        return self.img

    def gather_data(self, w=None, h=None):
        if w is None or h is None:
            w = self.width//5
            h = self.height//5
        self.grab_screen()
        self.process_img(w, h)
        keys = get_pressed_keys()
        output = self.keys_to_output(keys)
        return self.img, output


class AbstractDriver(abc.ABC):

    def __init__(self):
        self.same_key_pressed_counter = 0

    def go_straight(self, full_speed):
        ReleaseKey(A)
        ReleaseKey(D)
        if full_speed:
            PressKey(W)

    def go_left(self, full_speed):
        ReleaseKey(D)
        if full_speed:
            PressKey(W)
        PressKey(A)

    def go_right(self, full_speed):
        ReleaseKey(A)
        if full_speed:
            PressKey(W)
        PressKey(D)

    def stop(self):
        ReleaseKey(A)
        ReleaseKey(W)
        ReleaseKey(D)

    @abc.abstractmethod
    def drive(self, *args, **kwargs):
        pass
