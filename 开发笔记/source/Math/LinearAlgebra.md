# LinearAlgebra

## 平面的表示

### 点法式

**平面的一般式方程**：$Ax+By+Cz+D=0$

推导：设平面的法向量为$\vec n=(A,B,C)$，已知平面上一点$A(x_1, y_1,z_1)$，平面上的任一点为$R(x,y,z)$，易知$\vec {AR}·\vec n=0 $，则有
$$
\vec {AR} = (x-x_1,y-y_1,z-z_1) \\
{AR}·\vec n = A(x-x1)+B(y-y1)+C(z-z1) = 0 \\
Ax+By+Cz = Ax1+By1+Cz1 \\
Ax+By+Cz-(Ax1+By1+Cz1) = 0 \\
Ax+By+Cz+D=0
$$
前提回顾：法向量$\vec n$即一个与平面上的任意一个向量$\vec a$都垂直的向量，可表示为：$\vec n · \vec a=0$

## 直线的向量表示

直线方程（$P_1$为直线上一点，$\vec v$为直线的方向向量）：
$$
P_1+t\vec v =(x_1+tv_x, y_1+tv_y, z_1+tz_1)
$$
等价于：
$$
\left\{\begin{matrix} 
  x=x_1 + tv_x \\  
  y=y_1 + tv_y \\
  z=z_1 + tv_z \\
\end{matrix}\right.
$$


### 投影

设两个非零向量$\vec a$与$\vec b$的夹角为$\theta$，则$|\vec b|cos\theta$为向量$\vec b$在向量$\vec a$上的投影或标投影(scalar projection)
$$
|\vec b|cos\theta=\frac{a·b}{|a|}
$$

### 点到平面的距离

点到平面的距离公式：设平面外一点$M_1(x_1,y_1,z_1)$，平面的法向量$\vec n=(A,B,C)^T$，$M_1$到平面的距离为$\frac{Ax_1+By_1+Cz_1+D}{|A^2+B^2+C^2|}$

推导：取平面上任意一点$M(x,y,z)$，点$M_1$到平面的距离等价于$\vec {MM_1}$在平面法向量上的投影
$$
d = Prj_{\vec n}\vec {MM_1}=|\vec {MM_1}|cos\theta=\frac{\vec {MM_1}·\vec n}{|\vec n|} \\
=\frac{A(x_1-x)+B(y_1-y)+C(z_1-z)}{|\vec n|} \\
=\frac{Ax_1+By_1+Cz_1+D}{|\vec n|} \\
=\frac{Ax_1+By_1+Cz_1+D}{|A^2+B^2+C^2|}
$$


### 平面和线的交点

已知：

平面方程：
$$
Ax+By+Cz+D=0 \tag 1
$$
直线方程（$P_1$为直线上一点，$\vec v$为直线的方向向量）：
$$
P_1+t\vec v =(x_1+tv_x, y_1+tv_y, z_1+tv_z) \tag 2
$$
将(2)带入到(1)，得到**交点**$Q$
$$
A(x_1+tv_x)+B(y_1+tv_y)+C(z_1+tv_z)+D=0 \\
Ax_1+By_1+Cz_1+Atv_x+Btv_y+Ctv_z+D=0 \\
Ax_1+By_1+Cz_1+t(Av_x+Bv_y+Cv_z)+D=0 \\
t = -\frac{Ax_1+By_1+Cz_1+D}{Av_x+Bv_y+Cv_z} \\
t = -\frac{\vec n·\vec P_1+D}{\vec n·\vec v} \\
Q = P_1 - \frac{\vec n·\vec P_1+D}{\vec n·\vec v} \vec v
$$
<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/aUdHD.png" alt="enter image description here" style="zoom:67%;" />

特别地：如果是求**相机的射线和已知平面的交点**，则$P_1(0,0,0)$
$$
t = -\frac{D}{\vec n·\vec v} = -\frac{D}{Ax_1+By_1+Cz_1} \\
Q =  -\frac{D}{Ax_0+By_0+C*1} \vec v
$$
其中：
$$
\vec v = (x_0-0,y_0-0,1-0) = (x_0,y_0,1)
$$

## [平面在新系下的表示](https://math.stackexchange.com/questions/2502857/transform-plane-to-another-coordinate-system)

已知lidar系到camera系的TF变换为$M=T^C_L$，lidar系下的平面为：
$$
ax+by+cz+d=0
$$
求camera系下对该平面的描述：
$$
AX+BY+CZ+D=0
$$
平面写成矩阵形式则有：
$$
\pi^T X=0 \\
\pi^T M^{-1}MX=0 \\
(M^{-T}\pi)^{T}(MX)=0
$$
.. note:: 线性变换后直线是直线，平面还是平面（只是换个笛卡尔坐标系来描述这个平面）

camera系下的系数则为：$(M^{-T}\pi)=(A\ B\ C\ D)$

## 内积

- 将两个向量映射为一个实数
- $A·B=|A||B|cos\theta$：向量$B$的模为1时，该内积表示$A$向$B$所在直线投影的矢量长度

## 向量与基

给定一组基$\vec i,\vec j$，向量$A$可以表示为这组基的线性组合：
$$
A=x\vec i+y\vec j=(\vec i,\vec j)
\begin{pmatrix}
x \\
y
\end{pmatrix}
$$

## 线性变换的意义

### 基变换/坐标变换

向量在另一个系下的表征

- 将右边矩阵的每一个列向量样本数据，变换到左边矩阵**每一行行向量为基**所表示的空间中
- 将左边矩阵的每一个行向量样本数据，变换到右边矩阵**每一列列向量为基**所表示的空间中

### 相对运动/坐标系变换

系的相对运动

- e.g. 激光雷达系通过**旋转变换**（运动）到相机系，其欧拉角为（-90°，0，90°）

## 案例

[二维旋转矩阵](https://www.cnblogs.com/meteoric_cry/p/7987548.html)

对二维坐标系旋转之后，原来的点在旋转后的坐标系下的表示

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220311105915554.png" alt="image-20220311105915554" style="zoom:50%;" />

# 线性代数的几何直觉

## 向量

向量实际上是**基向量的线性组合**；线性组合时的**系数**即**坐标**；坐标也是**缩放量**。

如基向量为i和j，向量A=2i+3j，向量A的坐标即(2,3)
$$
A = \begin{bmatrix}
\hat i & \hat j 
\end{bmatrix}  
\begin{bmatrix}
2 \\ 3
\end{bmatrix} = 2 \hat i +  3 \hat j
$$
.. note:: 向量在左，坐标在右

.. note:: 一般字段/属性为行，数据为列，如点云[4,N]\(x,y,z,i)，又如：

![image-20220225140546559](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220225140546559.png)

## 矩阵

矩阵的几何直觉是运动（平移和旋转）

# 各种坐标系

## SLAM 14讲

### 描述规范

**坐标变换**：$a_1$为向量在坐标系1的描述；$a_2$为向量在坐标系2的描述；$R_{12}$：将向量从坐标系2转换到坐标系1下描述（同时也可以将其描述为坐标系1到坐标系2的旋转变换（坐标系变换））

## [ros坐标系](https://www.ros.org/reps/rep-0103.html)

### 旋转

- 固定坐标系(fix/extrinsic，外旋)：X->Y->Z
- 相对坐标系(intrinsic，内旋)：**Z->Y->X**，对应的角度为欧拉角

.. note:: 二者得到的旋转矩阵实际上一致（左固右相）

```python
from scipy.spatial.transform import Rotation
import numpy as np

lidar_frame = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
camera_frame = np.asarray([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
rotation_mat = camera_frame @ lidar_frame.T
rotation = Rotation.from_matrix(rotation_mat)
print(rotation.as_euler('ZYX', degrees=True)) # (-90,0,-90) 对应于 (zyx轴)
```

.. note:: 3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations

.. note:: 由此处得到的旋转矩阵描述的新坐标系在旧坐标系下的描述；

.. note:: 欧拉角的角度正负：按照右手定则，如**右手大拇指指向z-轴，四指弯曲的旋转方向**为α正方向

### ros静态TF

- ros的TF描述的是**坐标系的变换**，而不是**基坐标变换**

- 发布lidar->camera的python程序的launch文件

```xml
<launch>
    <!-- static_transform_publisher x y z yaw pitch roll 父坐标系(frame_id) 子坐标系(child_frame_id) -->
	<!-- ZYX: 使用的是内旋坐标系 -->
    <!-- 2：有两种解读，一种是go to，一种是relative to， -->
	<node pkg="tf2_ros" type="static_transform_publisher" name="lidar_2_camera" args="0, 0, 1, -1.570795, 0, -1.570795 lidar camera " />
</launch>
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220312094230162.png" alt="image-20220312094230162" style="zoom:67%;" />

- 发布lidar->camera的python程序

```python
#!/usr/bin/env python
import rospy

import tf2_ros
import geometry_msgs.msg


if __name__ == '__main__':
    rospy.init_node('tf2_publisher')
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        br = tf2_ros.TransformBroadcaster()

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "lidar"
        t.child_frame_id = "camera"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.0
        t.transform.rotation.x = -0.50
        t.transform.rotation.y = 0.50
        t.transform.rotation.z = -0.50
        t.transform.rotation.w = 0.50

        br.sendTransform(t)

    rospy.spin()
```

- 监听lidar->camera的CLI

![image-20220312101457181](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220312101457181.png)

```bash
# tf_echo <source_frame> <target_frame>  监听source_frame到target_frame的目标
$ rosrun tf tf_echo /lidar /camera
# At time 0.000
# - Translation: [0.000, 0.000, 1.000]
# - Rotation: in Quaternion [-0.500, 0.500, -0.500, 0.500]
#            in RPY (radian) [-1.571, -0.000, -1.571]
#            in RPY (degree) [-90.000, -0.000, -90.000]
```

- 监听lidar->camera的python程序

```python
import rospy
import tf2_ros

if __name__ == '__main__':
    rospy.init_node('tf2_echo_rospy')   
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            # 虽然API是target_frame, source_frame
            # 但实际调用应该是source_frame, target_frame
            trans = tfBuffer.lookup_transform("lidar", 'camera', rospy.Time())
            pass
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue           
        rate.sleep()
```

.. note::  `ros-tf-whoever-wrote-the-python-tf-api-f-ked-up-the-concept <https://www.hepeng.me/ros-tf-whoever-wrote-the-python-tf-api-f-ked-up-the-concept/>`_ 

![image-20220312104512404](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220312104512404.png)

### [描述规范](https://www.cnblogs.com/sxy370921/p/11726691.html)

- source、target frame是在**进行坐标变换**时的概念，source是坐标变换的源坐标系，target是目标坐标系。这个时候，这个变换代表的是**坐标变换**（数据的基变换）。
- parent、child frame是在**描述坐标系变换**时的概念，parent是原坐标系，child是变换后的坐标系，这个时候这个变换**描述的是坐标系变换**，也是child坐标系**在**parent坐标系下的描述。
- a frame到b frame的**坐标系变换**（frame transform），也表示了b frame在a frame的描述，也代表了把一个点在b frame里坐标变换成在a frame里坐标的**坐标变换**。
- 从parent到child的坐标系变换（frame transform）等同于把一个点从child坐标系向parent坐标系的**坐标变换**，等于child坐标系在parent frame坐标系的姿态描述。

## kitti数据集的坐标

- 除了论文之外，其测评工具([details](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d))的readme也提供了坐标系的相关信息，可以得出不同于ROS的TF，kitti提供的变换矩阵是基座标变换矩阵

![image-20220405235416211](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220405235416211.png)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220407102231534.png" alt="image-20220407102231534" style="zoom: 67%;" />

### 坐标系

- from tracking_devkit/cs_overview.pdf

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220410003806319.png" alt="image-20220410003806319" style="zoom: 50%;" />

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220410003528787.png" alt="image-20220410003528787" style="zoom: 50%;" />

- KITTI的航向角为$r_y$，物体在激光雷达系（pcdet的定义）下的航向角为$2\pi-(r_y+\frac{2}{\pi})=-(r_y+\frac{2}{\pi})$
- 只考虑航向角，将激光雷达系下的点转换到物体坐标系下：

$$
P_l=T_{lo}P_o=R_{lo}P_o+t_{lo} \\
$$

$$
P_o=R_{ol}(P_l-t_{lo}) \\

\begin{bmatrix}
cos\theta & sin\theta \\
-sin\theta & cos\theta
\end{bmatrix}
$$

.. note:: $R_{ol}$为向量从激光雷达系到物体坐标系的旋转变换（角度为物体系在激光雷达系下的航向角）

- 根据可视化的结果，pcdet预测出来的值为：物体系在激光雷达系下的航向角

  <img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220411151213052.png" alt="image-20220411151213052" style="zoom:67%;" />

为什么此处的物体坐标系不是相机系的表示（根据`tracking_devkit/cs_overview.pdf`）？

# Rotation

轴角、旋转向量

```python
import transforms3d
import numpy as np
from scipy.spatial.transform import Rotation

# 1. axis-angle representation
# 此处的axis不需要归一化
theta = np.pi / 4
print(transforms3d.axangles.axangle2mat([0, 0, 1], theta))
print(transforms3d.axangles.axangle2mat([0, 0, 4], theta))

# 2. rotation vector representation
# three-dimensional vector which is co-directional to the axis of rotation
# and whose norm gives the angle of rotation
# 该表征通过norm来蕴含旋转角的大小
print(Rotation.from_rotvec([0, 0, np.pi / 4]).as_matrix())

# 3. euler representation
print(Rotation.from_euler('XYZ', [0, 0, np.pi / 4]).as_matrix())
```
