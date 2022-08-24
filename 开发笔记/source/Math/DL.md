# DL

## loss

### CE

交叉熵损失描述的是两个概率分布的差异，以真值的概率为权，以**预测的概率的信息量**（概率的自然对数的负值）为待加权的对象；公式描述：$CE(P,Q)=E_{x\sim P} -logQ(x)$

```python
# 类别交叉熵损失（输入是未归一化的）
torch.nn.CrossEntropyLoss()
# 二元交叉熵损失（输入未经归一化的）
torch.nn.functional.binary_cross_entropy_with_logits()
torch.nn.BCEWithLogitsLoss()
# 二元交叉熵损失（输入已归一化）
torch.nn.BCELoss()
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220307152648271.png" alt="image-20220307152648271" style="zoom:50%;" />

.. note:: 类别交叉熵损失(category CE loss)和二元交叉熵损失(binary CE loss)，前者预测值对应的激活函数为softmax，后者预测值对应的激活函数sigmoid

.. note:: 分开计算softmax和log的效率会较低和不稳定

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220307142535643.png" alt="image-20220307142535643" style="zoom:50%;" />

### BCE

```python
loss = nn.BCEWithLogitsLoss()
input = torch.randn(1)
target = torch.Tensor([0.9])
output = loss(input, target)

print("正样本的预测概率: ", nn.Sigmoid()(input))
print("负样本的预测概率: ", 1 - nn.Sigmoid()(input))
print("正样本的信息熵:", -torch.log(nn.Sigmoid()(input)))
print("负样本的信息熵:", -torch.log(1 - (nn.Sigmoid()(input))))
print("二元交叉熵: ", output)
print("二元交叉熵(custom): ", -torch.log(nn.Sigmoid()(input)) * 0.9 +
                                    -torch.log(1 - (nn.Sigmoid()(input))) * 0.1
      )


# 正样本的预测概率: tensor([0.6628])
# 负样本的预测概率: tensor([0.3372])

# 正样本的信息熵: tensor([0.4112])
# 负样本的信息熵: tensor([1.0872])
# 二元交叉熵: tensor(0.4788)
# 二元交叉熵(custom): tensor([0.4788])
```

### smooth label+CE

$alpha$: 平滑参数；$K$: 类别参数

$y_{smooth}=(1-\alpha)y_{one\_hot}+\frac{\alpha}{K}$
$$
CEloss_{smooth}=-\Sigma y_{smooth}log{y_{predict}} \\
=-\Sigma [(1-\alpha)y_{one\_hot}+\frac{\alpha}{K}]log{y_{predict}} \\
= -\Sigma (1-\alpha)y_{one\_hot}log{y_{predict}}-\Sigma\frac{\alpha}{K}log{y_{predict}} \\
= (1-\alpha)CEloss-\Sigma\frac{\alpha}{K}log{y_{predict}}
$$

```python
import numpy as np
alpha = 0.1
one_hot = np.array([1, 0, 0])
y_smooth = (1 - alpha) * one_hot + alpha / len(one_hot)
print(y_smooth)

# [1 0 0] -> [0.93333333 0.03333333 0.03333333]


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, alpha=0.1):
        # 预测值的信息量
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1. - alpha) * nll_loss + alpha * smooth_loss
        return loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction    
    
from utils import LabelSmoothingCrossEntropy
criterion = LabelSmoothingCrossEntropy()
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

