import cv2
import numpy as np

mate_color_dict = {"purple": [[170, 210], [95, 135], [240, 255]],
                   "yellow": [[230, 255], [165, 210], [40, 100]],
                   }


def is_color(pixel, thresh):
    return thresh[2][1] >= pixel[0] >= thresh[2][0] and thresh[1][1] >= pixel[1] >= thresh[1][0] and \
           thresh[0][1] >= pixel[2] >= thresh[0][0]


def binary_thresh_pic(img, color_key):
    height, width, channel = img.shape
    bg = np.zeros((height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            if is_color(img[h, w, :], mate_color_dict[color_key]):
                bg[h, w] = 255
            else:
                bg[h, w] = 0
    return bg


imread = cv2.imread("../sample/2050448982.jpg")
h, w, c = imread.shape
imread0 = imread[int(0.8 * h):int(0.9 * h), w // 3: w // 2, :]

imread1 = imread[int(0.7 * h):int(0.8 * h), w * 2 // 3: w * 5 // 6, :]

thresh_pic = binary_thresh_pic(imread1, "yellow")

cv2.imshow("ss0", imread0)
cv2.imshow("ss1", imread1)
cv2.imshow("thresh_pic", thresh_pic)
cv2.waitKey(0)
