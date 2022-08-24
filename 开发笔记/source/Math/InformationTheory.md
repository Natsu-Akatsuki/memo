# InformationTheory

信息论为应数的一个分支。可用于量化signal里面有多少information。e.g. 应用于无线电通讯：解决在有噪声的通道中传输信息；应用于机器学习：描述概率分布中两个概率分布的相似性。

## 信息的量化和直觉

- 当我们observe到一件概率为1/e的事件时，我们就得到1 nats的信息量(information)

- 一件事情发生的概率很高，其包含的信息量就越低（e.g.明天太阳升起，information很低）
- 一件事情发生的概率很低，其包含的信息量就越高（e.g.明早有日食，information很高）

## Self Information

- （定义）数学描述：$I(x)=-log(P(x))$  概率的对数的负值

- 直觉：一件`随机事件`产生`某个结果`产生的information

- 拓展：熵越大，混沌程度越大，不确定性越大（但一开始其实跟热力学的熵没有关系）


## Shannon Entropy

- 将描述单个概率值/结果的信息量，拓展到描述整个概率分布/所有结果的信息量
- 数学描述：$H(X)=E(I(X))$，也记做$H(P)$
- 直觉：量化一个`概率分布` 的不确定性程度，表示self information的期望；self information的加权平均。
- 当随机变量连续时，`shannon entropy`也称为`differential entropy`

## KL divergence

- 直觉：描述`同一个随机变量`，两个不同概率分布的相似性（self information差值的期望）
- 数学定义：$D_{KL}(P||Q)=E_{X\sim{P}}{[log \frac{P(x)}{Q(x)}]}=E_{X \sim P}{[log{P(x)-log(Q(x)}]}$
- 以概率分布$P$的概率为权，对双方的**self-information的差值**进行加权平均
- 注意：$D_{KL}(P||Q)$ 和 $D_{KL}(Q||P)$是不一样的（asymmetric）
- 如果该随机变量的两个分布相同的话，则$KL$散度为0【所以有种可做损失函数的feel】

## cross-entropy

- 直觉：与$KL$散度相近，以概率分布$P$的概率为权对$Q(x)$的self information进行加权
- 公式描述：$H(P,Q)=E_{X\sim P} -logQ(x)$

- 在损失函数的实际中，以真值的概率为权，以预测的概率的信息量为待加权的对象