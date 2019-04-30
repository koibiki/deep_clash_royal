import cv2
import os
import numpy as np

mate_color_dict = {"blue_button": [[50, 110], [150, 200], [220, 255]],
                   "white": [[240, 255], [240, 255], [240, 255]],
                   "grey": [[200, 230], [200, 230], [200, 230]],
                   "red": [[200, 255], [40, 80], [40, 80]],
                   "blue": [[90, 110], [180, 210], [230, 255]],
                   }


def is_contain(rects):
    rects = [[rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]] for rect in rects]
    return (rects[0][0] < rects[1][0] and rects[0][1] < rects[1][1] and rects[1][2] < rects[0][2]
            and rects[1][3] < rects[0][3]) or \
           (rects[1][0] < rects[0][0] and rects[1][1] < rects[0][1] and rects[0][2] < rects[1][2]
            and rects[0][3] < rects[1][3])


def is_color(pixel, thresh):
    return thresh[2][1] >= pixel[0] >= thresh[2][0] and thresh[1][1] >= pixel[1] >= thresh[1][0] and \
           thresh[0][1] >= pixel[2] >= thresh[0][0]


def binary_thresh_pic(img, color_keys):
    height, width, channel = img.shape
    bg = np.zeros((height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            for i, color_key in enumerate(color_keys):
                if is_color(img[h, w, :], mate_color_dict[color_key]):
                    bg[h, w] = 255
                    break
                if i == len(color_keys) - 1:
                    bg[h, w] = 0
    return bg


img = cv2.imread("12573.jpg")
# img = cv2.resize(img, None, None, 0.5, 0.5)
h, w, c = img.shape

star_h = h // 4
clip_h = h // 3
star_w = w * 9 // 10
clip_w = w * 3 // 5

img = img[star_h: star_h + clip_h, star_w:, :]

# img = cv2.resize(img, None, None, 0.5, 0.5)

# h, w, c = img.shape

binary_thresh = binary_thresh_pic(img, ["red", "blue"])

kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(binary_thresh, kernel)

cv2.imshow("thresh", dilate)

image, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(box) for box in contours]

for rect in rects:
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)

rects = sorted(rects, key=lambda x: x[0])

if len(rects) == 2 and rects[0][0] < w / 6 and rects[0][1] > h * 2 / 3 and rects[1][0] > w / 2 and rects[1][1] > h / 2 \
        and rects[0][2] > w / 4:
    print("has_finish")

cv2.imshow("img", img)
cv2.waitKey(0)
