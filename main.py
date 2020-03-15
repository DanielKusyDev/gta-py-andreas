import numpy as np
import cv2
from PIL import ImageGrab
from numpy.linalg import lstsq

from directkeys import W, S, A, D

import pyautogui


class LaneFinder(object):
    def __init__(self, lines, window_sizes=(640, 480)):
        # assert isinstance(lines, cv2.HoughLinesP), "Lines is not a list instance."
        # assert len(lines) > 0, ?"Lines are empty."
        self.lines = lines
        self.lanes = []
        self.window_sizes = window_sizes

    def get_min_lines_y(self):
        y_min = self.window_sizes[1]
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            less_y = min(y1, y2)
            y_min = less_y if y_min > less_y else y_min
        return y_min

    def get_average_lane(self, lane):
        x1s = []
        y1s = []
        x2s = []
        y2s = []
        for data in lane:
            x1s.append(data[0])
            y1s.append(data[1])
            x2s.append(data[2])
            y2s.append(data[3])
        return int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))

    def calculate_lines_definitions(self):
        y_min = self.get_min_lines_y()
        y_max = self.window_sizes[1]
        line_definitions = []
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            x_coords = (x1, x2)
            y_coords = (y1, y2)
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, b = np.linalg.lstsq(A, y_coords)[0]
            x1 = (y_min - b) / m
            x2 = (y_max - b) / m
            points = [int(x1), y_min, int(x2), y_max]
            line_definitions.append([m, b, points])
        return line_definitions

    def get_lanes(self, abs_val=1.2):
        """ y = mx + b """
        try:
            line_definitions = self.calculate_lines_definitions()
            lanes = {}
            for line_def in line_definitions:
                tmp_lanes = lanes.copy()
                m, b, line = line_def
                # Initialize lanes dict lane
                if len(lanes) == 0:
                    lanes[m] = [line]
                else:
                    # Some lines can be duplicated
                    found_copy = False
                    for m, definition in tmp_lanes.items():
                        if not found_copy:
                            if abs(m * abs_val) > abs(m) > abs(m * (1 - abs_val)):
                                if abs(definition[0][1] * abs_val) > abs(b) > abs(definition[0][1] * (1 - abs_val)):
                                    lanes[m].append(line)
                                    found_copy = True
                                    break
                            else:
                                x=2
                                lanes[m] = [line]

            line_counter = {}
            for m, lane in lanes.items():
                line_counter[m] = len(lane)

            top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]
            if len(top_lanes) > 1:
                lane1_id, lane2_id = top_lanes
                self.lanes.append(self.get_average_lane(lanes[lane1_id]))
                self.lanes.append(self.get_average_lane(lanes[lane2_id]))
            else:
                self.lanes.append(self.get_average_lane(lanes[top_lanes[0][0]]))
            return self.lanes

        except Exception as e:
            print(e)


class ImageProcessing(object):

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.img = None
        # self.threshold1 = 120
        # self.threshold2 = 50
        self.threshold1 = 280
        self.threshold2 = 80
        self.minline = 220
        self.linethreshold = 75
        self.maxgap = 5
        self.tmp1 = 400
        self.tmp2 = 400
        self.k = None

    def show(self):
        self.img = np.array(ImageGrab.grab(bbox=(0, 40, self.width, self.height)))
        img = self.process_img()
        lines = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=self.linethreshold,
                                minLineLength=self.minline, maxLineGap=self.maxgap)
        self.draw_lanes(lines, img)
        cv2.imshow("ProcessedScreen", img)
        cv2.imshow("LanesDetectedScreen", self.img)
        self.handle_input()

    def _adjust(self, var1_name, var2_name, step: float = 10):
        value1 = getattr(self, var1_name)
        value2 = getattr(self, var2_name)
        if self.k == 2490368:
            setattr(self, var1_name, value1 + step)
        elif self.k == 2621440:
            setattr(self, var1_name, value1 - step)
        elif self.k == 2555904:
            setattr(self, var2_name, value2 + step)
        elif self.k == 2424832:
            setattr(self, var2_name, value2 - step)
        value1 = getattr(self, var1_name)
        value2 = getattr(self, var2_name)
        print(value1, value2)

    def handle_input(self):
        self.k = cv2.waitKeyEx(25)
        self._adjust("threshold1", "threshold2")
        # self._adjust("minline", "linethreshold")
        # self._adjust("tmp1", "tmp2", step=2)
        if self.k == 27:
            raise ValueError

    def roi(self, img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked = cv2.bitwise_and(img, mask)
        return masked

    def draw_lanes(self, lines, img):
        try:
            for l in lines:
                lane = l[0]
                cv2.line(img, (lane[0], lane[1]), (lane[2], lane[3]), [255, 255, 255], 10)
            lf = LaneFinder(lines, (self.width, self.height))
            lanes = lf.get_lanes()
            if lanes:
                for lane in lanes:
                    cv2.line(self.img, (lane[0], lane[1]), (lane[2], lane[3]), [0, 255, 0], 10)
        except Exception as e:
            print(str(e))

    def process_img(self):
        processed_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.Canny(processed_img, self.threshold1, self.threshold2)
        processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
        vertices = np.array([[0, 350], [0, 400], [700, 400], [300, 170]])
        processed_img = self.roi(processed_img, [vertices])
        return processed_img


if __name__ == "__main__":
    image_processing = ImageProcessing(640, 480)
    while True:
        try:
            image_processing.show()
        except ValueError:
            break
