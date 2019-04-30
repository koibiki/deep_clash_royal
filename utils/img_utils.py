import cv2
import numpy as np

mate_color_dict = {"purple": [[100, 180], [10, 100], [100, 181]],
                   "white": [[230, 255], [230, 255], [230, 255]],
                   "2": [[65, 100], [55, 90], [20, 45]]
                   }


def is_color(pixel, thresh):
    return thresh[2][1] >= pixel[2] >= thresh[2][0] and thresh[1][1] >= pixel[1] >= thresh[1][0] and \
           thresh[0][1] >= pixel[0] >= thresh[0][0]


def thresh_pic(img, color_key):
    height, width, channel = img.shape
    bg = np.zeros((height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            if is_color(img[h, w, :], mate_color_dict[color_key]):
                bg[h, w] = 255
            else:
                bg[h, w] = 0
    return bg
