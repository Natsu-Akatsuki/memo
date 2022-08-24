# IPM

## 坐标系

相机坐标系：x下，y右，z前

世界坐标系

路面坐标系

车体坐标系



- [ ] 相关坐标系的整理
- [ ] 公式的合并问题
- [ ] 需要解决的问题

## 成像公式

**公式一：**
$$
\begin{bmatrix}
u \\
v \\
w 
\end{bmatrix}  = \begin{bmatrix}
p_{11} & p_{12} & p_{13} \\
p_{21} & p_{22} & p_{23} \\
p_{31} & p_{32} & p_{33} 
\end{bmatrix} \begin{bmatrix}
X \\
Y \\
Z 
\end{bmatrix} + \begin{bmatrix}
t_1\\
t_2 \\
t_3 
\end{bmatrix}
$$


世界系下一点$(X,Y,Z)^T$在成像平面的坐标$(x,y)^T$为：
$$
x = \frac{u}{w} = \frac{p_{11}X+p_{12}Y+p_{13}Z+t_1}{p_{31}X+p_{32}Y+p_{33}Z+t_1} \\
y = \frac{v}{w} =\frac{p_{21}X+p_{22}Y+p_{23}Z+t_1}{p_{31}X+p_{32}Y+p_{33}Z+t_1}
$$

## 逆射影变换

### Part One

已知新的约束（映射点在平面$aX+bY+cZ+d=0$上），则公式一可以拓展到**公式二**：
$$
w\begin{bmatrix}
x \\
y \\
1 \\
0
\end{bmatrix}  = \begin{bmatrix}
p_{11} & p_{12} & p_{13} & 0 \\
p_{21} & p_{22} & p_{23} & 0 \\
p_{31} & p_{32} & p_{33} & 0 \\
a & b & c & d 
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}
+
\begin{bmatrix}
t_1 \\
t_2 \\
t_3 \\
0
\end{bmatrix}
$$

$$
w\begin{bmatrix}
x \\
y \\
1 \\
0
\end{bmatrix}  = \begin{bmatrix}
p_{11} & p_{12} & p_{13} & 0 \\
p_{21} & p_{22} & p_{23} & 0 \\
p_{31} & p_{32} & p_{33} & 0 \\
a & b & c & 0 
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}
+
\begin{bmatrix}
t_1 \\
t_2 \\
t_3 \\
d
\end{bmatrix}
$$

左右挪位：
$$
\begin{bmatrix}
-t_1 \\
-t_2 \\
-t_3 \\
-d
\end{bmatrix}
= \begin{bmatrix}
p_{11} & p_{12} & p_{13} & 0 \\
p_{21} & p_{22} & p_{23} & 0 \\
p_{31} & p_{32} & p_{33} & 0 \\
a & b & c & 0 
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}
- w\begin{bmatrix}
x \\
y \\
1 \\
0
\end{bmatrix}
$$

合并：
$$
\begin{bmatrix}
-t_1 \\
-t_2 \\
-t_3 \\
-d
\end{bmatrix}
= \begin{bmatrix}
p_{11} & p_{12} & p_{13} & -x \\
p_{21} & p_{22} & p_{23} & -y \\
p_{31} & p_{32} & p_{33} & -1 \\
a & b & c & 0 
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
w
\end{bmatrix}
$$
求逆变换，得**公式三**：
$$
\begin{bmatrix}
X \\
Y \\
Z \\
w
\end{bmatrix}
= \begin{bmatrix}
p_{11} & p_{12} & p_{13} & -x \\
p_{21} & p_{22} & p_{23} & -y \\
p_{31} & p_{32} & p_{33} & -1 \\
a & b & c & 0 
\end{bmatrix}^{-1}
\begin{bmatrix}
-t_1 \\
-t_2 \\
-t_3 \\
-d
\end{bmatrix}
$$

---

**NOTE**

- 该推导可以，已知地平面方程和地面的像素，恢复地面像素的三维信息
- 当平面为$z=0$时，公式三可拓展为：

$$
\begin{bmatrix}
X \\
Y \\
Z \\
w
\end{bmatrix}
= \begin{bmatrix}
p_{11} & p_{12} & p_{13} & -x \\
p_{21} & p_{22} & p_{23} & -y \\
p_{31} & p_{32} & p_{33} & -1 \\
0 & 0 & 1 & 0 
\end{bmatrix}^{-1}
\begin{bmatrix}
-t_1 \\
-t_2 \\
-t_3 \\
0
\end{bmatrix}
$$

---

### Part Two

从左到右依次是：像素点的齐次坐标，相机内参矩阵，相机外参矩阵（$T=[R|t]$，描述了相机在世界系下的位姿），世界系下的一点

$$
\begin{bmatrix}
u \\
v \\
w \\
\end{bmatrix}
= K_{3×3}T_{3×4}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}
$$
其中：像素点的坐标为$x=\frac {u}{w}$，$y=\frac {v}{w}$

逆透视变换，**公式四**：
$$
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}
= (K_{3×3}T_{3×4})^{-1}\begin{bmatrix}
u \\
v \\
w \\
\end{bmatrix}
$$
由于实际的过程中是使用像素坐标$(x,y)$，是丢失了深度信息$w$的，因此无法用公式四去恢复点的三维信息



# 方案

将FV/PV（前视图/透视图 perspective image）转换到BEV（俯视图/鸟瞰图）的方案有如下几种：

## 根据四组及以上的匹配点进行IPM

根据至少四组的匹配点求取射影矩阵，难点在于**在原图和目标图上寻找匹配点**

## 已知地平面模型的IPM

为简化描述以下坐标系只涉及相机系和成像平面，追加额外的系也只是多一些TF变换而已

### 模型一：Y=-h

已知：地面点在成像平面的像素坐标$u,v$；

假设：地面为平面其在相机系下的表征为$Y=-h$（$h$为相机距离地面的高度）

求解：可复原地面点$(X_c,Y_c,Z_c)$的在相机系下的三维坐标$(X_c,0,Z_c)$
$$
Z_c\begin{bmatrix}
u \\
v \\
1 \\
\end{bmatrix}
= \begin{bmatrix}
f_xX_c+c_xZ_c \\
f_yY_c+c_yZ_c \\
Z_c \\
\end{bmatrix}
= \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1 \\

\end{bmatrix}_{内参矩阵}

\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
\end{bmatrix}
$$
<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220302132611748.png" alt="image-20220302132611748" style="zoom:67%;" />

两个未知数，两个方程可解：
$$
Z_cu = f_xX_c+c_xZ_c \\
\\
Z_cv = f_yY_c+c_yZ_c \\ 
Z_cv= -f_yh+c_yZ_c \\
$$

$$
Z_c = \frac{-f_yh}{v-c_y} \\
X_c = \frac{Z_c(u-c_x)}{f_x} \tag{1.1}
$$

矩阵表示（$R$为内参矩阵）：
$$
Z_cR^{-1}\begin{bmatrix}
u \\
v \\
1 \\
\end{bmatrix}
= 
\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
\end{bmatrix}
$$


### 模型二：Ax+By+Cz+D=0

已知：地面点在成像平面的像素坐标$u,v$；

假设：地面为平面其在相机系下的表征为$aX+bY+cZ+d=0$（可通过点云分割+TF得到相机系下的地面方程）

求解：可复原在相机系下的地面点$(X_c,Y_c,Z_c)$
$$
Z_c\begin{bmatrix}
u \\
v \\
1 \\
0 \\
\end{bmatrix}
= \begin{bmatrix}
f_xX_c+c_xZ_c \\
f_yY_c+c_yZ_c \\
Z_c \\
aX_c+bY_c+cZ_c+d \\

\end{bmatrix}
= \begin{bmatrix}
f_x & 0 & c_x & 0 \\
0 & f_y & c_y & 0 \\
0 & 0 & 1 & 0 \\
a & b & c & d \\
\end{bmatrix}_{内参矩阵+地面约束}

\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
1 \\
\end{bmatrix}
$$
三个未知数，三个方程，可解：
$$
Z_c=\frac{-(aX_c+BY_c+D)}{c} \\
$$
矩阵表示为：
$$
Z_cR^{-1}\begin{bmatrix}
u \\
v \\
1 \\
\end{bmatrix}
= 
\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
\end{bmatrix}
$$

### Q&A

Q: 这种方法的效果会在什么时候会有问题？

地面点并不在预期平面模型上

Q: 相关的代码








$$
Z_c\begin{bmatrix}
u \\
v \\
1 \\
w \\
\end{bmatrix}
= \begin{bmatrix}
f_xX_c+c_xZ_c \\
f_yY_c+c_yZ_c \\
Z_c \\
aX_c+bY_c+cZ_c+d \\

\end{bmatrix}
= \begin{bmatrix}
f_x & 0 & c_x & 0 \\
0 & f_y & c_y & 0 \\
0 & 0 & 1 & 0 \\
a & b & c & d \\
\end{bmatrix}_{内参矩阵+地面约束}

\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
1 \\
\end{bmatrix}
$$



$$
Z_c\begin{bmatrix}
u \\
v \\
1 \\
\end{bmatrix}
= \begin{bmatrix}
f_xX_c+c_xZ_c \\
f_yY_c+c_yZ_c \\
Z_c \\
\end{bmatrix}
= \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1 \\

\end{bmatrix}_{内参矩阵}

\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
\end{bmatrix}
$$

$$
Z_cu = f_xCx+Z_c
$$

