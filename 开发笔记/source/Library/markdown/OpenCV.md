# OpenCV

## 使用摄像头

- example1

```python
import cv2
capture = cv2.VideoCapture(0)

# VideoCaptureProperties
capture.set(3, 1280)  # 常用配置属性，宽
capture.set(4, 720)    # 高
capture.set(5, 30)      # 帧率
while (True):
    ret, frame = capture.read()                           
    cv2.imshow('frame', frame)
    # return the Unicode code point for a one-character string.
    if cv2.waitKey(1) == ord('q'):
        break
```

- example2

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

## [两张图片的叠放](https://blog.csdn.net/fanjiule/article/details/81607873)

- 营造图层叠放效果

```python
import cv2
# 加权系数、偏置项
add_img =  cv2.addWeighted(img_1, 0.7, img_2, 0.3, 0)
```

- 掩膜操作
- [图片水平拼接](https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/)

```python
 image_grey = cv2.hconcat([image_l_ir, image_r_ir])
```

## 判断点是否在某个多边形中

```python
import cv2
# 轮廓点、测试点、是否返回距离(ture：表示该点在多边形中)
left_sign = cv2.pointPolygonTest(contour_, test_point, False)
# 其返回值是浮点型
```

## 图片读写和显示

```python
cv2.namedWindow("窗口名称")
# 读图片
img = cv2.imread(image_path)
# 显示图片
cv2.imshow("窗口名称", img)
# + 限定尺寸大小(W,H)
cv2.imshow('窗口名称', cv2.resize(img, dsize=(600, 320)))
```

## 视频流

- 生成视频流

```python
for split, dataset in zip(splits, datasets):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # 编码方式
    vout = cv2.VideoWriter(<"输出的文件名">, fourcc , 30.0, (img_w, img_h))
    for i, data in enumerate(tqdm.tqdm(loader)):
        ...            
        vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
        vout.write(vis)

    vout.release()
```

- [读写视频流](https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/)

## 窗口

```python
# 定义窗口名称
cv2.namedWindow("窗口名称")
cv2.destroyAllWindows()
```

## 通道转换

```python
# 颜色通道/空间变换
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

## 按键

```python
key = cv2.waitKey(1)
if key & 0xFF == ord('q'):
    break
```

## Calibration

### 去畸变

```python
distortion = np.loadtxt("畸变系数txt文件")
intrinsic_matrix = np.loadtxt("内参矩阵")
# 消除图像distortion
img = cv2.undistort(img, intrinsic_matrix, distortion)

# 消除图像点的畸变（如果没有指定P，那么将输出归一化坐标点）
img = cv2.undistortPoints(src=points, cameraMatrix=self.intri_matrix, distCoeffs=distortion, P=intrinsic_matrix)
```

### Q&A

- 为什么`projectPoints`含有畸变系数？

> 其得到的是考虑了畸变的二维图像点

- 在计算重投影误差时，由于真值考虑上了畸变模型（opencv检测到的棋盘格2D点），则三维的投影点也需要考虑上畸变模型。

```python
# 使用opencv 3D->2D的接口的话需要去一次畸变才是自己写的重投影
re_projection, _ = cv2.projectPoints(corners_world[:, :3].astype(np.float32), rvec, tvec, self.intri_matrix, self.distor)
undistor_img_cv = cv2.undistortPoints(src=np.squeeze(re_projection), cameraMatrix=self.intri_matrix,
                                                  distCoeffs=self.distor,
                                                  P=self.intri_matrix)

undistor_img_ours, _ = self.lidar_to_img(corners_world[:, :3])
```

## 添加元素

### 加圆

- 给定中心位置和半径画实心或空心圆

```python
photo = cv2.imread('<图形路径>')
cv2.circle(photo, center=(500, 400), radius=100, color=(0, 0, 255), thickness=2)

# 可视化2D的投影点云
for (x, y), c in zip(pts_2d, color):
    # 图片，圆心位置位置，圆半径，圆颜色，边界厚度（-1：填充）
    cv2.circle(img, (x, y), 1, [c[2], c[1], c[0]], -1)
```

### 加文字

```python
cv2.putText(img, <文本>, <位置>, cv2.FONT_HERSHEY_SIMPLEX, fontHeight=0.5, color(255, 0, 255), thickness=1)
```

## 交互操作

```python
# ret: tuple(four element)
ROI = cv2.selectROIs(img, fromCenter=False, showCrosshair=True)
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220208163816121.png" alt="image-20220208163816121" style="zoom:67%;" />

## colormap

``` python
import cv2
im = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
imC = cv2.applyColorMap(im, cv2.COLORMAP_JET)
```

## Smooth

```python
def dilate(range_img):
    kernel = np.ones((3, 3), dtype=np.uint8)
    range_img = cv2.dilate(range_img, kernel, iterations=3)
    return range_img

```

### 高斯滤波

```python
img = cv2.GaussianBlur(img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
```

## Q&A

- xkbcommon: ERROR: failed to add default include path .../anaconda3/envs/.../share/X11/xkb

> 添加环境变量 XKB_CONFIG_ROOT=/usr/share/X11/xkb

- [qt.qpa.plugin:Could not load the Qt platform plugin “xcb“](https://blog.csdn.net/LOVEmy134611/article/details/107212845)

