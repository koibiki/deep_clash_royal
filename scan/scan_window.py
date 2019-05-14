from airtest.core.api import *
from airtest.core.android.minicap import *

import cv2
import numpy as np

device = connect_device("Android://127.0.0.1:5037/cd9faa7f")
cap = Minicap(device.adb, [1080, 1920])  # 分辨率可以适当修改，提高速度

index = 0
while True:
    image_bytes = cap.get_frame_from_stream()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, None, 0.5, 0.5)
    cv2.imshow("window", img)
    cv2.waitKey(1)
    print(index)
    index += 1
