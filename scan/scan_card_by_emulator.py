import cv2
import os.path as osp
import os

from device.emulator import Emulator


def save_card(img, i, name):
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

    root0 = "../../card/" + name + "/0"
    root1 = "../../card/" + name + "/1"
    root2 = "../../card/" + name + "/2"
    root3 = "../../card/" + name + "/3"
    os.makedirs(root0, exist_ok=True)
    os.makedirs(root1, exist_ok=True)
    os.makedirs(root2, exist_ok=True)
    os.makedirs(root3, exist_ok=True)

    cv2.imwrite(root0 + "/0_" + str(i * 4) + ".jpg", img0)
    cv2.imwrite(root1 + "/0_" + str(i * 4 + 1) + ".jpg", img1)
    cv2.imwrite(root2 + "/0_" + str(i * 4 + 2) + ".jpg", img2)
    cv2.imwrite(root3 + "/0_" + str(i * 4 + 3) + ".jpg", img3)


if __name__ == '__main__':
    host_id = "127.0.0.1:62001"
    guest_id = "127.0.0.1:62025"
    host_device = Emulator(host_id, "one")
    guest_device = Emulator(guest_id, "two")

    i = 0
    while True:
        host_frame, host_state_code = host_device.get_frame()
        if host_state_code == 1 and host_frame is not None:
            save_card(host_frame, i, 'host')
            i += 1

        guest_frame, guest_state_code = guest_device.get_frame()
        if guest_state_code == 1 and guest_frame is not None:
            save_card(guest_frame, i, 'guest')
            i += 1
