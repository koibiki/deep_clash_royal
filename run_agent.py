import cv2

from game.clash_royal import ClashRoyal
from utils.c_lib_utils import convert2pymat

if __name__ == '__main__':

    address = "http://127.0.0.1:35013/device/cd9faa7f/video.flv"
    # address = "./scan/s1.mp4"
    capture = cv2.VideoCapture(address)
    cv2.namedWindow('image')

    i = 0

    root = "../vysor/"

    clash_royal = ClashRoyal(root)

    while True:

        state, img = capture.read()

        if state:
            if i % 5 != 0:
                i = i + 1
                continue

            img = cv2.resize(img, (540, 960))

            pymat = convert2pymat(img)

            result = clash_royal.frame_step(img)

            cv2.imshow('image', img)
            cv2.waitKey(1)
            i += 1
        else:
            break

    cv2.destroyAllWindows()
