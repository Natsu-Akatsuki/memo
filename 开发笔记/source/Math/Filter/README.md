# Filter

滤波即加权(weighting)，比如低通滤波就是对低频量的权值设为1，对高频量的权值设为0。

## 卡尔曼滤波

### 加权与权值

卡尔曼滤波就是观测值和预测值进行加权。其中观测值可以从传感器获取，预测值可以从上一时刻的状态预测而来。其中的权值是动态变化的，称为卡尔曼增益。**卡尔曼增益越大**（一般来说，**预测值的不确定性越大**），则测量值的权值越大，越相信**测量值是准的**；越小则对预测值的权值越大，越相信预测值是准的。这种权是基于不确定性来获取的。

### 三种假设

1. 传感器的**测量噪声**，状态转移的**过程噪声**需要服从零均值1方差的高斯分布
2. 线性系统
3. 马尔科夫假设：当前时刻的状态只与上一时刻的状态有关

### 公式

- 状态量和其协方差矩阵

$$
\begin{equation} \label{eq:statevars} 
\begin{aligned} 
\mathbf{\hat{x}}_k &= \begin{bmatrix} 
\text{position}\\ 
\text{velocity} 
\end{bmatrix}\\ 
\mathbf{P}_k &= 
\begin{bmatrix} 
\Sigma_{pp} & \Sigma_{pv} \\ 
\Sigma_{vp} & \Sigma_{vv} \\ 
\end{bmatrix} 
\end{aligned} 
\end{equation}
$$

- 预测方程/状态转移方程（假定匀速运动模型）和预测矩阵(**prediction matrix**) $\mathbf{F}_k$

$$
\begin{split} 
\color{deeppink}{p_k} &= \color{royalblue}{p_{k-1}} + \Delta t &\color{royalblue}{v_{k-1}} \\ 
\color{deeppink}{v_k} &= &\color{royalblue}{v_{k-1}} 
\end{split}
$$

$$
\begin{align} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \begin{bmatrix} 
1 & \Delta t \\ 
0 & 1 
\end{bmatrix} \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \\ 
&= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \label{statevars} 
\end{align}
$$

其协方差矩阵为：
$$
\begin{equation} 
\begin{split} 
Cov(x) &= \Sigma\\ 
Cov(\color{firebrick}{\mathbf{A}}x) &= \color{firebrick}{\mathbf{A}} \Sigma \color{firebrick}{\mathbf{A}}^T 
\end{split} \label{covident} 
\end{equation}
$$
总的预测方差和新的协方差矩阵如下：
$$
\begin{equation} 
\begin{split} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \\ 
\color{deeppink}{\mathbf{P}_k} &= \mathbf{F_k} \color{royalblue}{\mathbf{P}_{k-1}} \mathbf{F}_k^T 
\end{split} 
\end{equation}
$$

$$
\begin{equation} 
\begin{aligned} 
\mathbf{H}_k \color{royalblue}{\mathbf{\hat{x}}_k’} &= \color{fuchsia}{\mathbf{H}_k \mathbf{\hat{x}}_k} & + & \color{purple}{\mathbf{K}} ( \color{yellowgreen}{\vec{\mathbf{z}_k}} – \color{fuchsia}{\mathbf{H}_k \mathbf{\hat{x}}_k} ) \\ 
\mathbf{H}_k \color{royalblue}{\mathbf{P}_k’} \mathbf{H}_k^T &= \color{deeppink}{\mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^T} & – & \color{purple}{\mathbf{K}} \color{deeppink}{\mathbf{H}_k \mathbf{P}_k \mathbf{H}_k^T} 
\end{aligned} \label {kalunsimplified} 
\end{equation}
$$

- 观测方程：建模为$\mathbf{H}_k$，model_state=$\mathbf{H}_k$(sensor_reading)，其新的均值和协方差为：

$$
\begin{equation} 
\begin{aligned} 
\vec{\mu}_{\text{expected}} &= \mathbf{H}_k \color{deeppink}{\mathbf{\hat{x}}_k} \\ 
\mathbf{\Sigma}_{\text{expected}} &= \mathbf{H}_k \color{deeppink}{\mathbf{P}_k} \mathbf{H}_k^T 
\end{aligned} 
\end{equation}
$$

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/gauss_13.jpg" alt="gauss_12" style="zoom: 33%;" />

- 观测值：含如下均值和协方差$\color{yellowgreen}{(\mu_1}, \color{mediumaquamarine}{\Sigma_1}) = (\color{yellowgreen}{\vec{\mathbf{z}_k}}, \color{mediumaquamarine}{\mathbf{R}_k})$

### 实例

- 两个一维高斯分布的乘积构成了一个新的高斯分布：

$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma_f}e^{-\frac{(x-\mu_f)^2}{2\sigma^2_f}} \\ 
g(x) = \frac{1}{\sqrt{2\pi}\sigma_g}e^{-\frac{(x-\mu_g)^2}{2\sigma^2_g}} \\

f(x)g(x) = \frac{1}{2\pi\sigma_f\sigma_g}e^{-\frac{(x-\mu_f)^2}{2\sigma^2_f}} e^{-\frac{(x-\mu_g)^2}{2\sigma^2_g}}
$$

这个高斯分布经过整合之后其均值和方差为：（写成这种形式是因为可以提取出一个卡尔曼增益的权值，其中**卡尔曼增益越大，测量值的权值**越大，反之越小）
$$
\mu'=\mu_0 + \frac {\sigma_0^2(\mu_1-\mu_0)}{\sigma_0^2+\sigma_1^2} = \mu_0 + k(\mu_1-\mu_0) = (1-k)\mu_0+k\mu_1
\\
\sigma'^2=\sigma_0^2 - \frac {\sigma_0^4}{\sigma_0^2+\sigma_1^2}=\sigma_0^2-k\sigma_0^2=(1-k)\sigma_0^2 \\
k = \frac {\sigma_0^2}{\sigma_0^2+\sigma_1^2}
$$
<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220423100907128.png" alt="image-20220423100907128" style="zoom:67%;" />

矩阵形式：
$$
{\mathbf{K}} = \Sigma_0 (\Sigma_0 + \Sigma_1)^{-1}  \\
{\vec{\mu}’} = \vec{\mu_0} + {\mathbf{K}} (\vec{\mu_1} – \vec{\mu_0})=\mathbf{K}\vec{\mu_1}-(1-\mathbf{K})\vec{\mu_0}\\\ 
{\Sigma’} = \Sigma_0 – {\mathbf{K}} \Sigma_0=  (1-{\mathbf{K}})\Sigma_0
$$
.. note:: 此处的$\mu_1$为测量值在状态空间的投影值

#### 设计一个基于3D bbx的卡尔曼滤波

- 定义一个KalmanFilter类，定义属性：状态量、状态转移矩阵、观测量；定义方法：更新、预测、获取当前的状态

### Q&A

- 测量值的方差为0代表什么？

测量值不具有不确定性，数据准确无误，没有噪声。

- [为什么SORT/AB3DMOT这些的状态转移矩阵没有时间](https://github.com/xinshuoweng/AB3DMOT/issues/62)/[Sort的速度含义](https://github.com/abewley/sort/issues/59)

对于AB3DMOT的速度，个人理解倾向于将这个状态量描述为0.1s之内x/y/z方向上的位移量（m/0.1s），到时将速度状态量乘以10即可

### 拓展资料

- [实例：AB3MOT](https://github.com/xinshuoweng/AB3DMOT/blob/master/AB3DMOT_libs/kalman_filter.py)(commit ID: 1f3e202f91983ac8d52ed40029961de51209a0a7)
- [工具：线性方程->矩阵](https://ww2.mathworks.cn/help/symbolic/sym.equationstomatrix.html;jsessionid=c415b75da64c71352a2dd892e85a)
- [how-a-kalman-filter-works-in-pictures](http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mathybits)