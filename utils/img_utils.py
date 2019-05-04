import numpy as np
from scipy import random

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


# 定义添加椒盐噪声的函数
def add_salt_and_pepper(src, percetage):
    SP_NoiseImg = src
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randX = random.random_integers(0, src.shape[0] - 1)
        randY = random.random_integers(0, src.shape[1] - 1)
        if random.random_integers(0, 1) == 0:
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
