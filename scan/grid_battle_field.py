import cv2

num_align_width = 7
num_align_height = 8

img = cv2.imread("../sample/field.jpg")

# img = cv2.resize(img, None, None, 0.5, 0.5, )
h, w, c = img.shape

w_gap = h_gap = w // num_align_width

offset_w = w_gap // 2

img_copy = img.copy()
for x in range(num_align_width):
    for y in range(num_align_height):
        # cv2.line(img, (0, (y + 1) * h_gap), (w, (y + 1) * h_gap), (200, 200, 255), thickness=1)
        # cv2.line(img, (x * w_gap, 0), (x * w_gap, num_align_height * h_gap), (200, 200, 255), thickness=1)

        cv2.line(img_copy, (0, (y + 1) * h_gap + h_gap // 4), (w, (y + 1) * h_gap + h_gap // 4), (200, 200, 255),
                 thickness=1)
        cv2.line(img_copy, (offset_w + x * w_gap, 0), (offset_w + x * w_gap, num_align_height * h_gap), (200, 200, 255),
                 thickness=1)
        if x != num_align_width - 1:
            cv2.circle(img_copy, (offset_w + x * w_gap + w_gap // 2, (y + 1) * h_gap + h_gap * 3 // 4), 2, (0, 0, 255),
                       2)

field = img[h_gap + h_gap // 4: 9 * h_gap + h_gap // 4, h_gap // 2:-h_gap // 2, :]

field = cv2.resize(field, (384 // 2, 512 // 2))

f_h, f_w, f_c = field.shape
field = cv2.flip(field, 1)
img_copy = cv2.flip(img_copy, 1)
cv2.imshow("ss", field)
cv2.imshow("ssc", img_copy)
cv2.waitKey(0)
# cv2.imwrite(osp.join(save_dir, "s0_" + img_name), img)
# os.rename(img_path, osp.join(root, "s0_" + img_name))
