import ctypes
import cv2
import numpy as np

from utils.c_lib_utils import convert2pymat

ll = ctypes.cdll.LoadLibrary
lib = ll("./libc_opencv.so")
# lib.test.restype = ctypes.c_float
# frame = cv2.imread('test.jpg')
# frame_data = np.asarray(frame, dtype=np.uint8)
# frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
# value = lib.test(frame.shape[0], frame.shape[1], frame_data)

# arr = np.zeros((3, 5))


arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
tmp = np.asarray(arr, dtype=np.uint8)
rows, cols = tmp.shape
dataptr = tmp.ctypes.data_as(ctypes.c_char_p)
# lib.show_uchar_matrix(dataptr, rows, cols)

# arr = np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.uint8)
# tmp = np.asarray(arr, dtype=np.uint8)
#
# tmp = cv2.imread("full_finish.jpg", cv2.IMREAD_GRAYSCALE)
#
# rows, cols = tmp.shape
# dataptr = tmp.ctypes.data_as(ctypes.c_char_p)
# lib.transfer_image(dataptr, rows, cols, 1)

pyarray = [1., 2., 3., 4., 5.1]
carray = (ctypes.c_float * len(pyarray))(*pyarray)
lib.sum_array.restype = ctypes.c_float
sum = lib.sum_array(carray, len(pyarray))
print(sum)
lib.change_array(carray, len(pyarray))
print(np.array(carray))

tmp = cv2.imread("full_finish.jpg")

pymat = convert2pymat(tmp)

lib.transfer_mat(pymat)
