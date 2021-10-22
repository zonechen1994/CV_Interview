

## 标签平滑-$label$ $smoothing$

在深度学习样本训练的过程中，当我们采用 $one$-$hot$ 标签去进行计算交叉熵损失时，只考虑到训练样本中正确的标签位置（$one$-$hot$ 标签为 $1$ 的位置）的损失，而忽略了错误标签位置（$one$-$hot$ 标签为 $0$ 的位置）的损失。这样一来，模型可以在训练集上拟合的很好，但由于其他错误标签位置的损失没有计算，导致预测的时候，预测错误的概率增大。为了解决这一问题，标签平滑的正则化方法便应运而生。

### 什么是标签平滑

标签平滑采用如下思路：**在训练时即假设标签可能存在错误，避免“过分”相信训练样本的标签。当目标函数为交叉熵时，这一思想有非常简单的实现，称为标签平滑（$Label$ $Smoothing$）。** 在训练样本中，我们并不能保证所有的样本标签都标注正确，如果某个样本的标注是错误的，那么在训练时，该样本就有可能对训练结果产生负面影响。一个很自然的想法是，如果我们有办法“告诉”模型，样本的标签不一定正确，那么训练出来的模型对于少量的样本错误就会有“免疫力”。

为了达到这个目标，我们很自然想到的方法是：在每次迭代时，并不直接将(${x_i}$,${y_i}$)放入训练集，而是设置一个错误率 $\epsilon$，以 1-$\epsilon$ 的概率将(${x_i}$,${y_i}$)代入训练，以 $\epsilon$ 的概率将(${x_i}$,1-${y_i}$)代入训练。这样，模型在训练时，既有正确标签输入，又有错误标签输入，可以想象，如此训练出来的模型不会“全力匹配”每一个标签，而只是在一定程度上匹配。这样，即使真的出现错误标签，模型受到的影响就会更小。

那么，这是否意味着我们在每次训练时，都需要对标签进行随机化处理呢？答案是否定的，我们有更好的方法来解决，也就是标签平滑。下面我们介绍标签平滑的具体思路。

### 标签平滑的推理

假设(${x_i}$,${y_i}$)是训练集的一个样本。当我们采用交叉熵来描述损失函数时，对于每一个样本 $i$，损失函数为：

$$
L_i = -y_iP(\tilde{y_i}=1|x_i)-(1-y_i)P(\tilde{y_i}=0|x_i)
$$

经过随机化之后，新的标签有 $1$-$\epsilon$ 的概率与 ${y_i}$ 相同，有 $\epsilon$ 的概率不同（即 $1$-${y_i}$）。所以，采用随机化的标签作为训练数据时，损失函数有 1-$\epsilon$ 的概率与上面的式子相同，有 $\epsilon$ 的概率为：

$$
L_i = -(1-y_i)P(\tilde{y_i}=1|x_i)-y_iP(\tilde{y_i}=0|x_i)
$$

我们把上面两个式子按概率加权平均，就可以得到：

$$
L_i = -[\epsilon(1-y_i)+(1-\epsilon)y_i]P(\tilde{y_i}=1|x_i)-[\epsilon{y_i}+(1-\epsilon)(1-y_i)]P(\tilde{y_i}=0|x_i)
$$

为了简化上面的式子，我们令：

$$
y^{`} = \epsilon(1-y_i)+(1-\epsilon)y_i
$$

综合一下，就可以得到：

$$
L_i = -{y_i}^{`}P(\tilde{y_i}=1|x_i)-(1-{y_i}^{`})P(\tilde{y_i}=0|x_i)
$$

这个式子和原先的交叉熵表达式相比，只有 ${y_i}$ 被换成了${y_i}^{`}$，其他的内容全部都没有变。这实际上等价于：**把每个标签 ${y_i}$ 换成了 ${y_i}^{`}\$，然后再进行正常的训练过程。** 因此，我们并不需要在训练前进行随机化处理，只需要把每个标签替换一下即可。

那么为什么我们说这个过程就可以把标签平滑掉呢？我们可以从下面的式子里看出：

$$
y_{i}^{\prime}=\left\{\begin{aligned}
\epsilon, y_{i} &=0 \\
1-\epsilon, y_{i} &=1
\end{aligned}\right.
$$

什么意思呢，就是说当标签为 $0$ 的时候，我们并不是将 $0$ 直接放入模型中训练，而是将其换成一个比较小的数字 $\epsilon$，同样地，如果标签为 $1$，我们也将其替换为较接近的数 $1-{\epsilon}$。

也就是说我们告诉模型，$“1”$ 不一定为真，$“0”$ 不一定为假。为了方便看出效果，我们可以给出交叉熵模型的表达式：

$$
y_i = \frac{1}{1+e^{-w^{T}x_i}}
$$

由此可见，在交叉熵模型中，模型输出永远不可能达到 $0$ 和 $1$，因此模型会不断增大 $w$，使得预测输出尽可能逼近 $0$ 或 $1$，而这个过程与正则化是矛盾的，或者说，有可能出现过拟合。如果我们把标签 $0$ 和 $1$ 分别替换成 ${\epsilon}$ 和 $1$-${\epsilon}$，模型的输出在达到这个值之后，就不会继续优化。因此，所谓平滑，指的就是把两个极端值 $0$ 和 $1$ 变成两个不那么极端的值。下面我们再举一个实际的例子说明。

### 实际举例分析

假设有一批样本，样本类别总数为 $5$，从中取出一个样本，得到该样本的 $one$-$hot$ 化后的标签为[0,0,0,1,0]，假设我们已经得到了该样本进行 $softmax$ 的概率矩阵 $p$ ，即：

$$
p = [p_1, p_2, p_3, p_4, p_5] = [0.1, 0.1, 0.1, 0.36, 0.34]
$$

则我们使用未经过标签平滑的数据根据交叉熵求得当前的 $loss_1$ 为：

$$
loss = -(0*log0.1 + 0*log0.1 + 0*log0.1 + 1*log0.36 + 0*log0.34)
$$

计算结果为：

$$
loss_1 = -log0.36
$$

**可以发现没有标签平滑计算的损失只考虑正确标签位置的损失，而不考虑其他标签位置的损失，** 这就会出现一个问题，即不考虑其他错误标签位置的损失，这会使得模型过于关注增大预测正确标签的概率，而不关注减少预测错误标签的概率，最后导致的结果是模型在自己的训练集上拟合效果非常良好，而在其他的测试集结果表现不好，即过拟合，也就是说模型泛化能力差。

那么我们再来看一下使用标签平滑后的结果。我们知道标签平滑的公式为：

$$
y^{`}=(1-\epsilon)*y+\epsilon*(1-y)
$$

还是上面那组数据，假设平滑因子 $\epsilon$=$0.1$，将数据中的 $y$ 进行如下变化：

$$
y_1 = (1-\epsilon)*[0,0,0,1,0]=[0,0,0,0.9,0]
$$

$$
y_2=\epsilon[1,1,1,1,1]=[0.1,0.1,0.1,0.1,0.1]
$$

$$
y = y_1+y_2=[0.1,0.1,0.1,1,0.1]
$$

因此 $y$ 就是我们经过平滑操作后得到的标签，接着我们就可以求出平滑后该样本的交叉熵损失 $loss_2$ 为：

$$
loss_2 = -(0.1*log0.1 + 0.1*log0.1 + 0.1*log0.1 + 1*log0.36 + 0.1*log0.34)
$$

很明显我们可以看出 $loss_2$ 是大于 $loss_1$ 的。并且平滑过后的样本交叉熵损失就不仅考虑到了训练样本中正确的标签位置（$one$-$hot$ 标签为 $1$ 的位置）的损失，也稍微考虑到其他错误标签位置（$one$-$hot$ 标签为 $0$ 的位置）的损失，导致最后的损失增大，导致模型的学习能力提高，即要下降到原来的损失，就得学习的更好，也就是迫使模型往增大正确分类概率并且同时减小错误分类概率的方向前进。

下面我们给出在使用标签平滑时的 $softmax$ 损失的代码实现：

```python
def cross_entropy_loss(preds, target, reduction):
    logp = F.log_softmax(preds, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

# one-hot编码
def onehot_encoding(labels, n_classes):
    return torch.zeros(labels.size(0), n_classes).to(labels.device).scatter_(
        dim=1, index=labels.view(-1, 1), value=1)

def label_smoothing(preds, targets,epsilon=0.1):
	#preds为网络最后一层输出的logits
	#targets为未one-hot的真实标签
    n_classes = preds.size(1)
    device = preds.device

    onehot = onehot_encoding(targets, n_classes).float().to(device)
    targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(
        device) * epsilon / n_classes
    loss = cross_entropy_loss(preds, targets, reduction="mean")
    return loss
```

### 总结

在几乎所有的情况下，使用标签平滑训练可以产生更好的校准网络，能够告诉”模型，样本的标签不一定正确，那么训练出来的模型对于少量的样本错误就会有“免疫力”，从而更好地去泛化网路，最终对不可见的生产数据产生更准确的预测。

### 引用

- https://zhuanlan.zhihu.com/p/101553787
- https://juejin.cn/post/6844903520089407502
- https://www.cnblogs.com/whustczy/p/12520239.html
- https://blog.csdn.net/Matrix_cc/article/details/105344967
- https://blog.csdn.net/qq_43211132/article/details/100510113
- https://blog.csdn.net/qq_44015059/article/details/109479164?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-4&spm=1001.2101.3001.4242



