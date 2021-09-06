import cv2
from pathlib import Path
import numpy as np

# row_inner_corners_num = 7
# column_inner_corners_num = 5
# img_path = Path()
# img_path = str(img_path.resolve('.') / 'cv2_common_utils' / 'sample/checkboard.bmp')
# img_cv2 = cv2.imread(img_path)
# img_cv2_grey = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
# ret, corners_img = cv2.findChessboardCornersSB(img_cv2_grey, (7, 5), None)
# vis = cv2.drawChessboardCorners(img_cv2, (7, 5), corners_img, ret)
#
# corners_3d = np.zeros((row_inner_corners_num * column_inner_corners_num, 3), np.float32)
# corners_3d[:, :2] = np.mgrid[0:column_inner_corners_num, 0:row_inner_corners_num].T.reshape(-1, 2)
#
# cv2.imshow('grey', img_cv2)
# cv2.waitKey(0)
if __name__ == '__main__':

    row_inner_corners_num = 7
    column_inner_corners_num = 5
    img_path = Path()
    img_path = '/mnt/disk2/dataset/iidcc_sample/1611500404.4752169/images/000000.png'
    # img_path = '/mnt/disk2/dataset/000000.png'
    img_cv2 = cv2.imread(img_path)
    img_cv2_grey = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    ret, corners_img = cv2.findChessboardCornersSB(img_cv2_grey, (7, 5), None)
    vis = cv2.drawChessboardCorners(img_cv2, (7, 5), corners_img, ret)

    cv2.imshow('grey', vis)
    cv2.waitKey(0)
# 得到标定版的角点的位置
'''
def findChessboardCornersSB(image, patternSize, corners=None, flags=None):  # real signature unknown;

    """
    findChessboardCornersSB(image, patternSize[, corners[, flags]]) -> retval, corners

@param patternSize. 元组，格子的个数（看中间的，先row再column） Number of inner corners per a chessboard row and column
@return retval{bool}
@return corners{np(H*W,1,2)}
    """

e.g.
ret, corners = cv2.findChessboardCornersSB(gray, (6, 5), None)
'''

# 可视化结果
'''
def drawChessboardCorners(image, patternSize, corners, patternWasFound):
    """
    drawChessboardCorners(image, patternSize, corners, patternWasFound) -> image
    摘要：可视化检测出来的角点

@param image Destination image. It must be an 8-bit color image. 源图
@param patternSize Number of inner corners per a chessboard row and column(patternSize = cv::Size(points_per_row,points_per_column)).
@param corners Array of detected corners, the output of findChessboardCorners.
@param patternWasFound Parameter indicating whether the complete board was found or not. The return value of findChessboardCorners should be passed here.The function draws individual chessboard corners detected either as red circles if the board was not found, or as colored corners connected with lines if the board was found.
    """

e.g.
vis = cv2.drawChessboardCorners(vis, (row, column), corners, ret < bool >)
'''

# 相机内外参标定
'''
def calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs=None, tvecs=None, flags=None,
                    criteria=None):  # real signature unknown; restored from __doc__
    """
    calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]) -> 
    retval, cameraMatrix, distCoeffs, rvecs, tvecs
    .   @overload
    """
    pass
'''
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, corners_img, gray.shape[::-1], None, None)
