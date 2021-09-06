import cv2
import numpy as np

intrinsic = np.asarray([[1948.624, 0, 980.2519],
                        [0, 1947.808, 529.5462],
                        [0, 0, 1]], dtype=np.float32)
distor = np.asarray([-0.54801, 0.30312, 0, 0, -0.08168], dtype=np.float32)

img_np = cv2.imread('/home/ah_chung/WIN_20210123_09_45_12_Pro.jpg')
img_undistor = cv2.undistort(img_np, intrinsic, distor)
cv2.imshow("undis", img_undistor)
cv2.waitKey(0)