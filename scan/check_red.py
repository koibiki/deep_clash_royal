import cv2
import numpy as np
import os
from utils.img_utils import thresh_pic, extract_attention

# root = "F:\\gym_data\\clash_royal\\win\\718912023\\running"
#
#
# pattern = cv2.imread(root + "\\0.jpg")
#
# for i in range(len(os.listdir(root))):
#     img = cv2.imread(root + "\\" + str(i) + ".jpg")
#     masked = extract_attention(img, pattern)
#     cv2.imshow("masked", masked)
#     cv2.waitKey(0)

capture = cv2.VideoCapture("../env/test2.mp4")

pattern = None

offset_w = 39
offset_h = 94
width = 1080 // 2 - offset_w * 2
height = 62 * 10

while capture.isOpened():
    ret, img = capture.read()
    if ret:
        img = cv2.resize(img, (540, 960))
        h, w = img.shape[:2]
        img = img[h * 3 // 7 + h // 100: h * 3 // 7 + h // 60, w // 4: w * 3 // 4, :]
        if pattern is None:
            pattern = img
        else:
            thresh = thresh_pic(img, ['split'])

            radio = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1]) / 255
            print(radio)
            masked = extract_attention(img, pattern)

            cv2.imshow("thresh", thresh)
            cv2.imshow("masked", masked)
            cv2.imshow("pattern", pattern)
            cv2.imshow("img", img)
            cv2.waitKey(0)
