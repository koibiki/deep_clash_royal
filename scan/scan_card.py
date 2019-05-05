import cv2
import os.path as osp
import os

capture = cv2.VideoCapture("../s2.mp4")

capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

i = 0
while True:
    state, img = capture.read()
    if state:

        h, w, c = img.shape

        start_w = int(0.23 * w)
        w_gap = int(0.022 * w)

        clip_w = w // 6

        start_h = h * 5 // 6
        clip_h = h * 10 // 85

        img0 = img[start_h: start_h + clip_h, start_w:start_w + clip_w, :]
        img1 = img[start_h: start_h + clip_h, start_w + clip_w + w_gap:start_w + 2 * clip_w + w_gap, :]
        img2 = img[start_h: start_h + clip_h, start_w + 2 * clip_w + 2 * w_gap:start_w + 3 * clip_w + 2 * w_gap, :]
        img3 = img[start_h: start_h + clip_h, start_w + 3 * clip_w + 3 * w_gap:start_w + 4 * clip_w + 3 * w_gap, :]

        cv2.imshow("0", img0)
        cv2.imshow("1", img1)
        cv2.imshow("2", img2)
        cv2.imshow("3", img3)
        cv2.waitKey(10)

        cv2.imwrite("../../card/s2/0/S1_" + str(i * 4) + ".jpg", img0)
        cv2.imwrite("../../card/s2/1/S2_" + str(i * 4 + 1) + ".jpg", img1)
        cv2.imwrite("../../card/s2/2/S3_" + str(i * 4 + 2) + ".jpg", img2)
        cv2.imwrite("../../card/s2/3/S4_" + str(i * 4 + 3) + ".jpg", img3)

        i += 1
    else:
        break
