import cv2
import numpy as np

mate_color_dict = {
    "red": [[200, 255], [170, 230], [170, 230]],
    "blue": [[170, 220], [170, 235], [200, 255]],
    "black": [[0, 50], [0, 50], [0, 50]]
}


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


imread = cv2.imread("G:\\\\PyCharmProjects\\\\deep_clash_royal\\\\sample\\\\84.jpg")

h, w, c = imread.shape

throne_start_w = w // 2 + 7
throne_end_w = throne_start_w + 5 + 50
opp_throne_start_h = h // 65
opp_throne_end_h = opp_throne_start_h + 15
mine_throne_start_h = h // 2 + 214
mine_throne_end_h = mine_throne_start_h + 15
opp_throne = imread[opp_throne_start_h:opp_throne_end_h, throne_start_w:throne_end_w, :]
cv2.imshow("opp", opp_throne)
cv2.imshow("opp_t", binary_thresh_pic(opp_throne, ["red", "black"]))
mine_throne = imread[mine_throne_start_h:mine_throne_end_h, throne_start_w:throne_end_w, :]
cv2.imshow("mine", mine_throne)

left_start_w = w // 5
left_end_w = left_start_w + 50
right_start_w = w * 3 // 5 + 68
right_end_w = right_start_w + 50
opp_start_h = h // 8 + 12
opp_end_h = opp_start_h + 15
mine_start_h = h // 2 + 109
mine_end_h = mine_start_h + 15
opp_left = imread[opp_start_h:opp_end_h, left_start_w:left_end_w, :]
cv2.imshow("left_opp", opp_left)
cv2.imshow("opp_t_l", binary_thresh_pic(opp_left, ["red", "black"]))
mine_left = imread[mine_start_h:mine_end_h, left_start_w:left_end_w, :]
cv2.imshow("left_mine", mine_left)
cv2.imshow("mine_t_l", binary_thresh_pic(mine_left, ["blue", "black"]))
opp_right = imread[opp_start_h:opp_end_h, right_start_w:right_end_w, :]
cv2.imshow("right_opp", opp_right)

cv2.imshow("o", imread)

cv2.waitKey(0)
