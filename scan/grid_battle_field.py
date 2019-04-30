import cv2
import os
import os.path as osp
from tqdm import *

num_align_width = 7
num_align_height = 10

img = cv2.imread("./0.jpg")

# img = cv2.resize(img, None, None, 0.5, 0.5, )
h, w, c = img.shape

w_gap = h_gap = w // num_align_width

img_copy = img.copy()
for x in range(num_align_width):
    for y in range(num_align_height):
        cv2.line(img, (0, (y + 1) * h_gap), (w, (y + 1) * h_gap), (200, 200, 255), thickness=1)
        cv2.line(img, (x * w_gap, 0), (x * w_gap, num_align_height * h_gap), (200, 200, 255), thickness=1)

        cv2.line(img_copy, (0, (y + 1) * h_gap + h_gap // 2), (w, (y + 1) * h_gap + h_gap // 2), (200, 200, 255), thickness=1)
        cv2.line(img_copy, (x * w_gap, 0), (x * w_gap, num_align_height * h_gap), (200, 200, 255), thickness=1)
cv2.imshow("ss", img)
cv2.imshow("ssc", img_copy)
cv2.waitKey(0)
# cv2.imwrite(osp.join(save_dir, "s0_" + img_name), img)
# os.rename(img_path, osp.join(root, "s0_" + img_name))
