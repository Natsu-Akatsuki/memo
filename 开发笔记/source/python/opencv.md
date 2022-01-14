 

[启动视频](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)

```python
import cv2
capture = cv2.VideoCapture(0
                           
capture.set(3, 1280)  # 常用配置属性，宽
capture.set(4, 720)    # 高
capture.set(5, 30)      # 帧率
while (True):
    ret, frame = capture.read()                           
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):   # return the Unicode code point for a one-character string.
        break
```

- [属性配置](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)



## [两张图片的叠放](https://blog.csdn.net/fanjiule/article/details/81607873)

- 营造图层叠放效果

```python
import cv2
# 加权系数、偏置项
add_img =  cv2.addWeighted(img_1, 0.7, img_2, 0.3, 0)
```

[掩膜操作](https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6)



## [判断点是否在某个多边形中](https://docs.opencv.org/4.5.0/d3/dc0/group__imgproc__shape.html#ga1a539e8db2135af2566103705d7a5722)

```python
import cv2
# 轮廓点、测试点、是否返回距离(ture：表示该点在多边形中)
left_sign = cv2.pointPolygonTest(contour_, test_point, False)
# 其返回值是浮点型
```





## 其他操作

```python
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
```



## 添加元素

### 加圆

```python
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
@param lineType Type of the circle boundary. See # 边界类型
@param shift Number of fractional bits in the coordinates of the center and in the radius value.
    """
e.g.
'''

photo = cv2.imread('<图形路径>')
cv2.circle(photo, center=(500, 400), radius=100, color=(0, 0, 255), thickness=2)

# 可视化2D的投影点云
for (x, y), c in zip(pts_2d, color):
    cv2.circle(img, (x, y), 1, [c[2], c[1], c[0]], -1)
```

### 加框

```python
void cv::rectangle 	( 	InputOutputArray  	img,
		Point  	pt1,
		Point  	pt2,
		const Scalar &  	color,
		int  	thickness = 1,
		int  	lineType = LINE_8,
		int  	shift = 0 
	) 		
Python:
	img	=	cv.rectangle(	img, pt1, pt2, color[, thickness[, lineType[, shift]]]	)
	img	=	cv.rectangle(	img, rec, color[, thickness[, lineType[, shift]]]	)
Draws a simple, thick, or filled up-right rectangle.

The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners are pt1 and pt2.

Parameters
    img	Image.
    pt1	Vertex of the rectangle.   对角点即可，不一定左上右下
    pt2	Vertex of the rectangle opposite to pt1 .
    color	Rectangle color or brightness (grayscale image).
    thickness	Thickness of lines that make up the rectangle. Negative values, like FILLED, mean that the function has to draw a filled rectangle.   (负数表示填充)
    lineType	Type of the line. See LineTypes
    shift	Number of fractional bits in the point coordinates. 
```

### 加文字

```python
'''
def putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None): # real signature unknown; restored from __doc__
    """
    putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
    摘要：添加文本，不能识别的将用？替代

@param img Image.  待加东西的图
@param text Text string to be drawn.  文本内容
@param org Bottom-left corner of the text string in the image. 文本左下角的坐标
@param fontFace Font type, see #HersheyFonts.
@param fontScale Font scale factor that is multiplied by the font-specific base size.
@param color Text color.
@param thickness Thickness of the lines used to draw a text.  # 字体粗细
@param lineType Line type. See #LineTypes
@param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
```



## 打开相机

```python
camera_open_flag = False
while not camera_open_flag:
    try:
        cap = cv2.VideoCapture(0)
        # 配置显示图片的宽、高、帧率
        cap.set(3, 1280)
        cap.set(4, 720)
        cap.set(5, 8)
        if cap.isOpened:
            print('successfully open camara')
            camera_open_flag = True
    except:
        time.sleep(1)
        print('retry to open the camera')
```



## 按键

```python
key = cv2.waitKey(1)
if key & 0xFF == ord('q'):
    break
```



## 标定

### 去畸变

```
distortion = np.loadtxt("畸变系数txt文件")
intrinsic_matrix = np.loadtxt("内参矩阵")
# 消除图像distortion
img = cv2.undistort(img, intrinsic_matrix, distortion)
```

