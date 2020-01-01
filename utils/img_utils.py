import numpy as np
import random
import cv2

from utils.func_call import func_time

mate_color_dict = {"purple": [[100, 180], [10, 100], [100, 181]],
                   "white": [[220, 255], [220, 255], [220, 255]],
                   "red": [[0, 34], [0, 34], [142, 178]],
                   "blue": [[120, 224], [80, 154], [0, 50]],
                   "split": [[65, 100], [65, 100], [225, 255]]}


# 18 52 80 154  155 196

def is_color(pixel, thresh):
    return thresh[2][1] >= pixel[2] >= thresh[2][0] and thresh[1][1] >= pixel[1] >= thresh[1][0] and \
           thresh[0][1] >= pixel[0] >= thresh[0][0]


def thresh_pic(img, color_keys):
    height, width, channel = img.shape
    bg = np.zeros((height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            for color_key in color_keys:
                if is_color(img[h, w, :], mate_color_dict[color_key]):
                    bg[h, w] = 255

    return bg

@func_time
def extract_attention(img, pattern, diff_thresh=10, area_thresh=100):
    np_abs = np.abs(img.astype(float) - pattern.astype(float))

    diff_value = np_abs if len(np_abs.shape) == 2 else np.mean(np_abs, axis=-1)
    diff_value_min = np_abs if len(np_abs.shape) == 2 else np.min(np_abs, axis=-1)

    diff_value[diff_value < diff_thresh] = 0
    diff_value[diff_value >= diff_thresh] = 255

    diff_value_min[diff_value_min < 35] = 0
    diff_value_min[diff_value_min >= 35] = 1

    diff_value = diff_value * diff_value_min

    mask = np.zeros(diff_value.shape, np.uint8)

    contours, hierarchy = cv2.findContours(diff_value.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > area_thresh:
            cv2.drawContours(mask, contours, i, 255, -1)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel)

    h, w = img.shape[:2]
    cell_w = w // 6
    cell_h = h // 8
    rect0 = ((cell_w * 2 + cell_w // 3, 0), (cell_w * 4 - cell_w // 3, cell_h))
    rect1 = ((cell_w - cell_w // 4, cell_h - cell_h // 2), (cell_w * 2 - cell_w // 3, cell_h * 2))
    rect2 = ((cell_w * 4 + cell_w // 3, cell_h - cell_h // 2), (cell_w * 5 + cell_w // 4, cell_h * 2))

    rect3 = ((cell_w * 2 + cell_w // 3, cell_h * 7 - cell_h // 3), (cell_w * 4 - cell_w // 3, cell_h * 8))
    rect4 = ((cell_w - cell_w // 4, cell_h * 6 - cell_h // 4), (cell_w * 2 - cell_w // 3, cell_h * 7))
    rect5 = ((cell_w * 4 + cell_w // 3, cell_h * 6 - cell_h // 4), (cell_w * 5 + cell_w // 4, cell_h * 7))
    rects = [rect0, rect1, rect2, rect3, rect4, rect5]

    for rect in rects:
        mask = cv2.rectangle(mask, rect[0], rect[1], 255, -1)

    masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    return masked


# 定义添加椒盐噪声的函数
def add_salt_and_pepper(src, percetage):
    SP_NoiseImg = src
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) == 0:
            SP_NoiseImg[randX, randY] = 0
        else:
            SP_NoiseImg[randX, randY] = 255
    return SP_NoiseImg


# 定义添加高斯噪声的函数
def add_gaussian_noise(image, percetage):
    G_Noiseimg = image
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(1, image.shape[0] - 2)
        temp_y = np.random.randint(1, image.shape[1] - 2)
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg
