import cv2
import os.path as osp
import os
from tqdm import *
import numpy as np

mate_color_dict = {"white": [[240, 255], [240, 255], [240, 255]], }


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


root = "../../vysor/fail"
save_dir = "../../num_data"
img_dirs = [osp.join(root, str(game_id) + "/running") for game_id in os.listdir(root)]

for img_dir in tqdm(img_dirs):
    game_id = img_dir.split("/")[-2]
    img_paths = [osp.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
    for img_name in os.listdir(img_dir):
        img_path = osp.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (540, 960))
        h, w, c = img.shape
        start_w = int(0.26 * w)

        start_h = int(h * 19.03 // 20)
        clip_h = h // 35
        clip_w = int(clip_h * 1.5)

        img0 = img[start_h: start_h + clip_h, start_w:start_w + clip_w, :]

        img0 = binary_thresh_pic(img0, "white")

        cv2.imshow("0", img0)

        cv2.waitKey(0)

        # cv2.imwrite(osp.join(save_dir, game_id + "_" + img_name), img0)
