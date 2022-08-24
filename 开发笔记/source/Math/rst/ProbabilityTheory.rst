ProbabilityTheory
=================

Chapter1: Count
----------------

Concept
^^^^^^^^


* 组合：n choose r, number of possible combinations of n objects taken r

.. math::

  \begin{pmatrix}
    n  \\
    r
  \end{pmatrix}

* 根据枚举可以论证，A实验如果有m种结果，B实验如果有n种结果，那么A和B实验一共有mn种结果

Combination
^^^^^^^^^^^^


组合描述了一个排列集合有多少组group（arragement的元素相同时则视为在同一个group，比如arrangement ``ABC`` 和 ``BCA`` 是属于同一个group）

Chapter2: Axioms of Probability
------------------------------------

Concept
^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 术语
     - 定义
     - —
   * - S: sample space
     - 所有可能的实验结果所组成的集合
     - experiment, all **possiable outcomes**
   * - E: event
     - 样本空间的子集
     - subset, sample space
   * - event occur
     - if the **outcome** of the experiment is contained in E, then we say that event E has occured
     - —


Axioms
^^^^^^^


* The probability of event is some number between 0 and 1.
* With probability 1, the outcome will be a point in the sample space.
* For any sequence of **mutablly exclusive events**\ , the probability of at least one of these events occuring is just the sum of their respective probabilities.

Chapter3: Conditional Probability
------------------------------------

Concept
^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 术语
     - 定义
   * - P(E|F): conditional probability
     - E occurs given F has occurred


Formula
^^^^^^^^

条件概率
~~~~~~~~

:math:`P(E|F)`: the event that E occurs given F has occured must be in EF, for the outcome is contained in E and in F

贝叶斯定理
~~~~~~~~~~

.. math::
  P(H|E)= \frac {P(E|H)P(H)}{P(E)}

贝叶斯定理描述的是一种概率更新。比如原本对某件事情的看法置信度是0.2，看到了一些新的证据后，对这件事的置信度更新为了0.5，贝叶斯定理就数学化了这种描述。

Chapter4: Random Variable
--------------------------------

概念
^^^^

.. list-table::
   :header-rows: 1

   * - 术语
     - 定义
   * - random variable
     - 定义在采样空间的实值函数 / 实验结果的函数


Chapter5: Continuous Random Variable
--------------------------------------------

*
  随机变量的分布函数 ≠ 概率分布

Code
----


* 计算一个服从正态分布变量的值（\ `scipy.stats.norm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html>`_\ ）

.. code-block:: python

   def get_pdf(x, mu, sigma):
       """
       Returns the value of the probability density function at x
       :param x:
       :param mu:
       :param sigma:
       :return:
       """
       from scipy.stats import norm
       import numpy as np

       # formula for pdf
       # pdf = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)

       return norm.pdf(x, mu, sigma)

Exercise
---------

条件概率
^^^^^^^^


* 有一盏灯（或红色或绿色），一传感器观测的准确率为90%，第一次开灯有等可能的概率出现红灯或者绿灯，问开灯后观测到红色的条件下，灯确实为红色的概率为：

.. math::
  P(灯观测结果为红色) = \frac{1}{2} × \frac{1}{10} + \frac{1}{2} * \frac{9}{10} = \frac{1}{2}  \\
  P(灯观测结果为红色，且实际也为红色) = \frac{1}{2} * \frac{9}{10} = \frac{9}{20} \\
  P(灯实际为红色|灯观测结果为红色) =  \frac{P(灯观测结果为红色，且实际也为红色)}{灯观测结果为红色} = 90\%

* 有一盏灯（或红色或绿色），一传感器观测的准确率为90%，第一次开灯有等可能的概率出现红灯或者绿灯，每进行一次观测后，灯的颜色有60%的概率转换为另一种灯的颜色，问第一次观测到红色的条件下，第二次依然观测到红色的概率：

.. math::
  P(第一次灯观测结果为红色) = \frac{1}{2} × \frac{1}{10} + \frac{1}{2} * \frac{9}{10} = \frac{1}{2}  \\

  P(TRTR) = \frac{1}{2} × \frac{9}{10} × \frac{4}{10} × \frac{9}{10} \\
  P(TRFG) = \frac{1}{2} × \frac{9}{10} × \frac{6}{10} × \frac{1}{10} \\
  P(FGFG) = \frac{1}{2} × \frac{1}{10} × \frac{4}{10} × \frac{1}{10} \\
  P(FGTR) = \frac{1}{2} × \frac{1}{10} × \frac{6}{10} × \frac{9}{10} \\

  P(两次灯观测结果均为红色) = 21.8\% \\
  P(两次灯观测结果均为红色|第一次灯观测结果为红色) = \frac{0.224}{0.5} = 43.6\%

* Joe有80%的置信度认为钥匙在自己口袋中，其中40%的置信度在自己左口袋，其中80%的置信度在自己右袋，问在左口袋没有找到钥匙的情况下，钥匙在其他口袋的概率？

.. math::
  P(左口袋没有钥匙) = 1 - \frac{4}{10} \
  P(左口袋没有钥匙，且钥匙在其他口袋) = \frac{4}{10}
  P(钥匙在其他口袋|左口袋没有钥匙) = \frac{P(左口袋没有钥匙，且钥匙在其他口袋)}{P(左口袋没有钥匙)} = \frac{2}{3}


* 一个硬币抛两次，在第一次头朝上的情况下，第二次为头朝上的概率？

.. math::
  P(第一次头朝上，且第二次头朝上) = \frac{1}{2} × \frac{1}{2} \
  P(第一次头朝上) = \frac{1}{2} \
  P(第一次头朝上，且第二次头朝上|第一次头朝上) = \frac{1}{2}

* 一个硬币抛两次，至少一次头朝上的情况下，第二次为头朝上的概率？

.. math::
  P(至少一次头朝上) = \frac{1}{2} × \frac{1}{2} + \frac{1}{2} × \frac{1}{2} + \frac{1}{2} × \frac{1}{2} \
  P(至少一次头朝上且第二次为头朝上) = \frac{1}{2} × \frac{1}{2} + \frac{1}{2} × \frac{1}{2} \
  P(第一次头朝上，且第二次头朝上|至少一次头朝上) = \frac{1}{3}

乘法公式
^^^^^^^^

Celine要选择一门课，她觉得化学得A的概率为$\frac{2}{3}$，法语课则为$\frac{1}{3}$，她通过投硬币觉得选哪一门课，问基于投硬币选课，她化学课得到A的概率？

.. math::
  P=\frac{1}{2}×\frac{2}{3}=\frac{1}{3}

全概率公式
^^^^^^^^^^

* 易受事故体质的发生事故的概率为0.4；不易受事故体质的发生事故的概率为0.2；人群中30%的人是易受事故体质的。一个人发生事故的概率是？

记 :math:`A` 为发生事故；:math:`B` 为易受事故体质；:math:`B_c` 为不易受事故体质

.. math::
  P(A)=P(A|B)P(B) + P(A|B_c)P(B_c) = 0.4×0.3 + 0.2×0.7 =0.26

贝叶斯定理
^^^^^^^^^^


* 易受事故体质的发生事故的概率为0.4；不易受事故体质的发生事故的概率为0.2；人群中30%的人是易受事故体质的。一个人如果发生了事故，那他是易受事故体质的概率是？

.. math::
  P(B|A)=\frac{P(A|B)P(B)}{P(A)}=\frac{0.4×0.3}{0.26} = \frac{6}{13}

* 一个糖尿病患者进行A test时出现阳性的概率为30%；有60%的置信度认为Jones患癌；Jones是一个糖尿病患者；Jones的A test结果为阳性？医生根据A test的结果得到的新的置信度是多少？

记A为患病；P为A test为阳性

.. math::
  P(A|P)=\frac{P(P|A)P(A)}{P(P)}=\frac{1×0.6}{1×0.6+0.3×0.4} = 0.833


概率分布
^^^^^^^^


* 两个\ **相互独立**\ 的正态分布，他们相加之后服从的分布是？

新的期望=期望和；新的方差=方差和


if:
  .. math::
    X \sim \mathcal{N}(\mu_X,\sigma^2_X)，Y \sim \mathcal{N}(\mu_X,\sigma^2_X) \\
then:
  .. math::
    X+Y \sim \mathcal{N}(\mu_X+\mu_Y,\sigma^2_X+\sigma^2_Y)
e.g.
  .. math::
    \mathcal{N}(1, 8) + \mathcal{N}(2, 8) = \mathcal{N}(3,16)

