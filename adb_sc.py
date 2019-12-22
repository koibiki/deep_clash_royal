import subprocess
import os
import cv2
import numpy as np

from utils.func_call import func_time


def check_screenshot():
    for i in range(4):
        if pull_screenshot(3 - i) is not None:
            return 3 - i


@func_time
def pull_screenshot(SCREENSHOT_WAY):
    if 1 <= SCREENSHOT_WAY <= 3:
        process = subprocess.Popen(
            'adb shell screencap -p',
            shell=True, stdout=subprocess.PIPE)
        image_bytes = process.stdout.read()
        if SCREENSHOT_WAY == 2:
            image_bytes = image_bytes.replace(b'\r\n', b'\n')
        elif SCREENSHOT_WAY == 1:
            image_bytes = image_bytes.replace(b'\r\r\n', b'\n')
        img = cv2.imdecode(np.fromstring(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (540, 960))
        return img
    elif SCREENSHOT_WAY == 0:
        screenshot_name = "screenshot.png"
        os.system('adb shell screencap -p /sdcard/{}'.format(screenshot_name))
        os.system('adb pull /sdcard/{} {}'.format(screenshot_name, 'screenshot.png'))
        img = cv2.imread('screenshot.png')
        return img
    else:
        return None


while True:
    image = pull_screenshot(2)
    cv2.imshow("img", image)
    cv2.waitKey(1)
