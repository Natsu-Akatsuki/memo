OpenCV
======

Calibration
***********

.. tabs::

   .. code-tab:: C++ C++

      //

   .. code-tab:: python Python

      distortion = np.loadtxt("畸变系数txt文件")
      intrinsic_matrix = np.loadtxt("内参矩阵")
      # 消除图像distortion
      img = cv2.undistort(img, intrinsic_matrix, distortion)

      # 消除图像点的畸变（如果没有指定P，那么将输出归一化坐标点）
      img = cv2.undistortPoints(src=points, cameraMatrix=self.intri_matrix, distCoeffs=distortion, P=intrinsic_matrix)


Q&A
^^^

Q1：为什么 ``projectPoints`` 含有畸变系数？

A1：其得到的是考虑了畸变的二维图像点

在计算重投影误差时，由于真值考虑上了畸变模型（opencv检测到的棋盘格2D点），则三维的投影点也需要考虑上畸变模型。

.. code-block:: python

   # 使用OpenCV 3D->2D的接口的话需要去一次畸变才是自己写的重投影
   re_projection, _ = cv2.projectPoints(corners_world[:, :3].astype(np.float32), rvec, tvec, self.intri_matrix, self.distor)
   undistor_img_cv = cv2.undistortPoints(src=np.squeeze(re_projection), cameraMatrix=self.intri_matrix,
                                                     distCoeffs=self.distor,
                                                     P=self.intri_matrix)

   undistor_img_ours, _ = self.lidar_to_img(corners_world[:, :3])


Camera
******
* 打开笔记本内置相机

.. tabs::

   .. code-tab:: C++ C++

      // TODO

   .. code-tab:: python Python

      import cv2

      capture = cv2.VideoCapture(0)

      # VideoCaptureProperties
      capture.set(3, 1280)  # width
      capture.set(4, 720)  # height
      capture.set(5, 30)  # frame rate
      cv2.namedWindow('frame', 0)
      while True:
         ret, frame = capture.read()
         cv2.imshow('frame', frame)
         # return the Unicode code point for a one-character string.
         if cv2.waitKey(1) == ord('q'):
            break
      cv2.destroyWindow('frame')

Element
*******

.. tabs::

   .. code-tab:: C++ C++

      //

   .. code-tab:: python Python

      # 一、给定中心位置和半径画实心或空心圆
      img = cv2.imread("图形路径")
      cv2.circle(img, center=(500, 400), radius=100, color=(0, 0, 255), thickness=2)

      # e.g. 可视化2D的投影点云
      for (x, y), c in zip(pts_2d, color):
         # 图片，圆心位置位置，圆半径，圆颜色，边界厚度（-1：填充）
         cv2.circle(img, (x, y), 1, [c[2], c[1], c[0]], -1)

      # 二、添加文本
      cv2.putText(img, "文本", <对应位置>, cv2.FONT_HERSHEY_SIMPLEX,
                        fontHeight=0.5, color(255, 0, 255), thickness=1)

      # 三、交互画ROI
      ROI = cv2.selectROIs(img, fromCenter=False, showCrosshair=True)

Geometry
************************

.. tabs::

   .. code-tab:: C++ C++

      #include <opencv2/core.hpp>
      #include <opencv2/imgproc.hpp>

      // 判断点是否在某个闭合轮廓中
      // region need float
      std::vector<cv::Point2f> region = {{452, 385}, {830, 385}, {393, 540}, {900, 540}};
      cv::Point2d bottom_middle_point = {452, 399};
      int ret = cv::pointPolygonTest(region, point, false);

   .. code-tab:: python Python


      # 判断点是否在某个闭合轮廓中
      # It returns positive (inside), negative (outside), or zero (on an edge) value
      # False: 不返回具体的距离仅返回（-1，0，1）
      sign = cv2.pointPolygonTest(contour, point, False)

Image
*****

.. tabs::

   .. code-tab:: C++ C++

      // 读取图片
      std::string file_name = "";
      cv::imread(file_name);

   .. code-tab:: python Python

      import cv2

      # 显示图片
      cv2.imshow("窗口名称", cv2.resize(img, dsize=(600, 320)))

      # 图层叠放（参数：加权系数、偏置项）
      add_img =  cv2.addWeighted(img_1, 0.7, img_2, 0.3, 0)

      # 图片拼接
      image = cv2.hconcat([image_l, image_r])

      # 颜色通道变换
      cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      # 伪彩色
      img = cv2.imread("文件名", cv2.IMREAD_GRAYSCALE)
      img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

Key
***

.. tabs::

   .. code-tab:: C++ C++

      //

   .. code-tab:: python Python

      key = cv2.waitKey(1) # unit: ms
      if key & 0xFF == ord('q'):
         break

Mat
***

- `CV_8UC1，CV_32FC3，CV_32S等参数的含义 <https://blog.csdn.net/Young__Fan/article/details/81868666>`_
- `Point2d <-> Point2f <https://answers.opencv.org/question/68215/how-do-i-cast-point2f-to-point2d/>`_

.. tabs::

   .. code-tab:: C++ C++

      // Point2f -> Point2d
      std::vector<cv::Point2f> vector1;
      std::vector<cv::Point2d> vector2;
      for (auto & point : vector1) {
         vector2.push_back(cv::Point2d((int) point.x, (int) point.y));
      }

   .. code-tab:: python Python

      # TODO

Smooth
******

.. code-block:: python

   # 一、腐蚀操作
   def dilate(img):
       kernel = np.ones((3, 3), dtype=np.uint8)
       img = cv2.dilate(img, kernel, iterations=3)
       return img

   # 二、高斯滤波
   img = cv2.GaussianBlur(img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)

Window
*******

.. tabs::

   .. code-tab:: C++ C++

      //

   .. code-tab:: python Python

      # 定义窗口名称
      cv2.namedWindow("窗口名称")

      // 销毁窗口
      cv2.destroyWindow('frame')
      cv2.destroyAllWindows()

Q&A
***


Q1：xkbcommon: ERROR: failed to add default include path .../anaconda3/envs/.../share/X11/xkb

A1：添加环境变量 XKB_CONFIG_ROOT=/usr/share/X11/xkb


Q2&A2：`qt.qpa.plugin:Could not load the Qt platform plugin “xcb“ <https://blog.csdn.net/LOVEmy134611/article/details/107212845>`_