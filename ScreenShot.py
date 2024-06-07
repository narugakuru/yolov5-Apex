import cv2
import numpy as np
from mss import mss

ScreenX = 2560
ScreenY = 1600
# ScreenX = 1920
# ScreenY = 1200
area = 320
half_area = area//2

window_size = (
    int(ScreenX / 2 - half_area),
    int(ScreenY / 2 - half_area),
    int(ScreenX / 2 + half_area),
    int(ScreenY / 2 + half_area)
)
Screenshot_value = mss()


def screenshot():
    img = Screenshot_value.grab(window_size)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img
