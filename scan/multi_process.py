from multiprocessing import Pool
import os, time, random
import numpy as np
import cv2

def long_time_task(index, item, item_shape, flag):
    print('Run task %s (%s)...' % (index, os.getpid()))
    start = time.time()
    print(str(flag))
    pic_shape = np.fromstring(item_shape, np.int32)
    pic = np.fromstring(item, np.uint8).reshape(pic_shape)
    cv2.imwrite("../sample/0_" + str(index) + ".jpg", pic)
    # print(pic)

    # time.sleep(random.random() * 3)
    end = time.time()
    print('Task %d runs %0.2f seconds.' % (index, (end - start)))


def apply(index, item, item_shape, flag):
    p.apply_async(long_time_task, args=(index, item, item_shape, flag))


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())

    item = cv2.imread("../sample/0.jpg")

    item_shape = np.array(item.shape, np.int32)
    item = item.tostring()
    item_shape = item_shape.tostring()

    p = Pool(4)
    for i in range(5):
        apply(i, item, item_shape, True)
    print('Waiting for all subprocesses done...')

    while True:
        pass

    print('All subprocesses done.')
