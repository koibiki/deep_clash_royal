import cv2
import os.path as osp
import os

capture = cv2.VideoCapture("./s6.mp4")

capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

i = 0
while True:
    state, img = capture.read()
    if state:
        img = cv2.resize(img, None, None, 0.5, 0.5)
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

        img0_ = cv2.resize(img0, None, None, 0.2, 0.2)
        img0_ = cv2.resize(img0_, (clip_w, clip_h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("0_", img0_)

        cv2.imshow("0", img0)
        cv2.imshow("1", img1)
        cv2.imshow("2", img2)
        cv2.imshow("3", img3)
        cv2.waitKey(0)

        # cv2.imwrite("../../card/s6/0/S6_" + str(i * 4) + ".jpg", img0)
        # cv2.imwrite("../../card/s6/1/S6_" + str(i * 4 + 1) + ".jpg", img1)
        # cv2.imwrite("../../card/s6/2/S6_" + str(i * 4 + 2) + ".jpg", img2)
        # cv2.imwrite("../../card/s6/3/S6_" + str(i * 4 + 3) + ".jpg", img3)

        i += 1
    else:
        break
