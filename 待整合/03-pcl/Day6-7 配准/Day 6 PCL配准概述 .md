# The PCL Registration API

https://pcl.readthedocs.io/projects/tutorials/en/latest/registration_api.html#registration-api

The problem of consistently aligning various 3D point cloud data views into a complete model is known as **registration**. Its goal is to find the relative positions and orientations of the separately acquired views in a global coordinate framework, such that the intersecting areas between them overlap perfectly. For every set of point cloud datasets acquired from different views, we therefore need a system that is able to align them together into a single point cloud model, so that subsequent processing steps such as segmentation and object reconstruction can be applied.

![_images/scans.jpg](https://pcl.readthedocs.io/projects/tutorials/en/latest/_images/scans.jpg)

A motivation example in this sense is given in the figure above, where a set of six individual datasets has been acquired using a tilting 2D laser unit. Since each individual scan represents only a small part of the surrounding world, it is imperative to find ways to register them together, thus creating the complete point cloud model as shown in the figure below.

![_images/s1-6.jpg](https://pcl.readthedocs.io/projects/tutorials/en/latest/_images/s1-6.jpg)

The algorithmic work in the PCL registration library is motivated by finding correct point correspondences in the given input datasets, and estimating rigid transformations that can rotate and translate each individual dataset into a consistent global coordinate framework. This registration paradigm becomes easily solvable if the point correspondences are perfectly known in the input datasets. This means that a selected list of points in one dataset have to “coincide” from a feature representation point of view with a list of points from another dataset. Additionally, if the correspondences estimated are “perfect”, then the registration problem has a closed form solution.

PCL contains a set of powerful algorithms that allow the estimation of multiple sets of correspondences, as well as methods for rejecting bad correspondences, and estimating transformations in a robust manner from them. The following sections will describe each of them individually.



# 配准概述

- 两帧点云的配准称为`pairwise registration`，它的输出为一个4×4的刚体变换。这个变换作用于`source`) 以实现跟`target` / `model`的对齐

- 配准步骤如下（一次迭代）

![_images/block_diagram_single_iteration.jpg](https://pcl.readthedocs.io/projects/tutorials/en/latest/_images/block_diagram_single_iteration.jpg)

> - 提取特征点来表征点云
> - 计算特征点的特征描述子（descriptors）
> - 基于特征描述子和坐标值获取特征点的相似度，找到特征点对（correspondences）
> - 假定数据存在噪声，则剔除有问题的特征点对
> - 使用特征点对进行配准



# 配准基本模块

## 特征点

特征点是特定场景下具有某种特别性质的点，如书的角点。常用的角点包括 NARF, SIFT and FAST. 



## 特征描述子

在得到特征点后，需要提取特征描述子的特征 [features](http://www.pointclouds.org/documentation/tutorials/how_features_work.php)，即得到一个特征向量. 常用的一些描述子包括：NARF, FPFH, BRIEF or SIFT.



## 获取特征对

根据不同类型的特征点，获取匹配对的方法有所不同：

对`point matching`(以点的x,y,z为特征，即特征在欧式空间下) 的方法（数据可无序/有序）

- brute force matching,
- kd-tree nearest neighbor search (FLANN),
- searching in the image space of organized data, and
- searching in the index space of organized data.

对`feature matching`的方法，只有以下方法可以使用

- 暴力匹配

- KD树



寻求对应的方法：

- 直接对A点云中的每个点在B点云中找对应
- 满足 “Reciprocal” correspondence estimation searches for correspondences from cloud A to cloud B, and from B to A and only use the intersection.



## 剔除错误的匹配对

This could be done using RANSAC or by trimming down the amount and using only a certain percent of the found correspondences.

A special case are one to many correspondences where one point in the model corresponds to a number of points in the source. These could be filtered by using only the one with the smallest distance or  by checking for other matchings near by.



## 估测变换矩阵

The last step is to actually compute the transformation.

- evaluate some error metric based on correspondence
- estimate a (rigid) transformation between camera poses (motion estimate) and minimize error metric
- optimize the structure of the points
- Examples: - SVD for motion estimate; - Levenberg-Marquardt with different kernels for motion estimate;
- use the rigid transformation to rotate/translate the source onto the target, and potentially run an internal ICP loop with either all points or a subset of points or the keypoints
- iterate until some convergence criterion is met



## 例程

### Iterative Closest Point

1. 找对应
2. 剔除不好的点云匹配对

3. 用好的匹配对数据估测刚体变换
4. 迭代



### Feature based registration

1. 提取关键点 e.g.  Keypoints (pcl::SIFT…something)
2. 计算描述子 e.g. FPFH descriptors (pcl::FPFHEstimation) at the keypoints （[教程](http://www.pointclouds.org/media/rss2011.html])）
3. 获取匹配对 pcl::CorrespondenceEstimation
4. 使用一个或多个 pcl::CorrespondenceRejectionXXX 方法剔除错误的匹配对
5. 用好的匹配对数据计算结果

