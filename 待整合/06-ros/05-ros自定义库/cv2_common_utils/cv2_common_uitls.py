# %%
# 去畸变
import numpy as np
import cv2

# 读图片
img = cv2.imread(image_path)
# 显示图片
cv2.imshow("窗口名称", img)
# + 限定尺寸大小(W,H)
cv2.imshow('窗口名称', cv2.resize(img, dsize=(600, 320)))

# 消除图像畸变
img_undistor = cv2.undistort(img, "内参矩阵", "畸变系数")

cv2.waitKey(0)
cv2.destroyAllWindows()

# 定义窗口名称
cv2.namedWindow("窗口名称")

# 颜色空间的变换
'''
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 转为灰度空间
@brief Converts an image from one color space to another.
'''

# 加圆
'''
def circle(img, center, radius, color, thickness=None, lineType=None, shift=None): # real signature unknown; restored from __doc__
    """
    circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
    摘要：给定中心位置和半径画实心或空心圆
    
@param img Image where the circle is drawn. 待加东西的图
@param center Center of the circle.  圆的中心
@param radius Radius of the circle.  圆的半径
@param color Circle color.           圆的颜色
@param thickness Thickness of the circle outline, if positive. Negative values, like #FILLED, mean that a filled circle is to be drawn.                               边界的厚度
@param lineType Type of the circle boundary. See #LineTypes
@param shift Number of fractional bits in the coordinates of the center and in the radius value.
    """
e.g.
photo = cv2.imread('<图形路径>')
cv2.circle(photo, center=(500, 400), radius=100, color=(0, 0, 255), thickness=2)
'''

# 加文字
'''
def putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None): # real signature unknown; restored from __doc__
    """
    putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
    摘要：添加文本，不能识别的将用？替代

@param img Image.  待加东西的图
@param text Text string to be drawn.  文本内容
@param org Bottom-left corner of the text string in the image. 左下角文本的坐标
@param fontFace Font type, see #HersheyFonts.
@param fontScale Font scale factor that is multiplied by the font-specific base size.
@param color Text color.
@param thickness Thickness of the lines used to draw a text.
@param lineType Line type. See #LineTypes
@param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
    """
    
e.g.
photo = cv2.imread('<图形路径>')
cv2.putText(photo, '{}'.format(文本内容), (100,200),
cv2.FONT_HERSHEY_SIMPLEX, <文字尺寸>1, <颜色>(0, 255, 0), <边界厚度> 1)
'''
# cv2.setMouseCallback(name, self.mouse_callback, param=(self,))

# 保存文件（需要将数据的通道进行转换）
img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
cv2.imwrite("<图片路径>", img_cv2)