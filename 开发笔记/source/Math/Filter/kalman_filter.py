import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
# https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

"""
Here is a filter that tracks position and velocity using a sensor that only reads position.
"""

# 步骤一：根据维度初始化卡尔曼滤波器
# dimension of state vector, measurement
f = KalmanFilter(dim_x=2, dim_z=1)

# 步骤二：给状态量赋初值（可以列向量或行向量）
f.x = np.array([[2.],  # position
                [0.]])  # velocity

# f.x = np.array([2., 0.])

# 步骤三：定义状态转移矩阵
f.F = np.array([[1., 1.],
                [0., 1.]])

# 步骤四：定义测量矩阵
# x(k) = position(k)
# v(k) = position(k) - position(k-1) / time
f.H = np.array([[1., 0.]])

# 步骤五：定义状态的协方差矩阵
# Here I take advantage of the fact that P already contains np.eye(dim_x), and just multiply by the uncertainty:
f.P *= 1000.
# 也可以：
# f.P = np.array([[1000.,    0.],
#                 [   0., 1000.] ])

# 步骤六：添加测量噪声
f.R = 5
# 也可以：
# f.R = np.array([[5.]])
# Note that this must be a 2 dimensional array, as must all the matrices.

# 步骤七：添加过程噪声
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

for i in range(100):
    # sensor data(position)
    z = 2
    # 用上一时刻的数据预测下一个时刻的状态
    f.predict()
    # 用传感器数据更新卡尔曼滤波器
    f.update(z)
    print(f.x)
