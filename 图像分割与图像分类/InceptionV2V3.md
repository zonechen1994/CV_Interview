## $Inception$ $V2/V3$

由于$Inception$ $V2/V3$是在同一篇论文中提出，所以在这里我们将这两个结构放在一起讲。其中$Inception$ $V2$主要是在$V1$的基础上加入$BN$层，这里不在做过多介绍，本文主要讲解$Inception$ $V3$。$Inception$ $V2/V3$设计之初主要针对的问题有：

($1$)减少特征的表征性瓶颈。直观上来说，当卷积不会大幅度改变输入维度时，神经网络可能会执行地更好。过多地减少维度可能会造成信息的损失，这也称为「表征性瓶颈」。

($2$)使用更优秀的因子分解方法，卷积才能在计算复杂度上更加高效。

#### $Inception$ $V2/V3$核心思想
**核心思想$1$：用两个$3$×$3$的卷积代替$5$×$5$的卷积**

大尺寸的卷积核可以带来更大的感受野，但也意味着更多的参数，比如$5$x$5$卷积核参数是$3$x$3$卷积核的$2$5$/$9$=$2.78$倍。为此，作者提出可以用$2个连续的$3$×$3$卷积层组成的小网络来代替单个的$5$x$5$卷积层，在保持感受野范围的同时又减少了参数量，如图1所示：

![图$1$](https://files.mdnice.com/user/15207/e90a6794-15f1-4a8f-965b-85aaaab43d22.png)

因此在原来的$Inception$ $V1$中的$5$×$5$卷积分解成两个$3$×$3$卷积，如下图$2$所示：


![图$2$](https://files.mdnice.com/user/15207/3ff6fad5-6e6d-4f8b-a2fe-80101ace2e6a.png)


**针对这种操作也许你会有一些疑问：** 

-这种替代会造成表达能力的下降吗？

作者在论文中做了大量实验可以表明不会造成表达缺失；

-$3$x$3$卷积之后还要再加激活吗？

作者也做了对比试验，表明添加非线性激活会提高性能。

**核心思想$2$：用$n$×$1$和$1$×$n$卷积代替$n$×$n$卷积**

从上面来看，大卷积核完全可以由一系列的小卷积核来替代，那能不能将小卷积分解的更小一点呢。在原论文中作者考虑到了$n$×$1$这种卷积如下图$3$所示：

![图$3$](https://files.mdnice.com/user/15207/9746fa7b-bfdd-4097-a001-3a90b295d1cd.png)

进一步的，作者将$n$x$n$的卷积核分解成$1$×$n$和$n$×$1$两个矩阵来进一步减少计算量，如下图$4$所示。实际上，作者发现在网络的前期使用这种分解效果并不好，只有在中度大小的$feature$ $map$上使用效果才会更好。（对于$m$×$m$大小的$feature$ $map$,建议$m$在12到20之间）。



![图$4$](https://files.mdnice.com/user/15207/382293c7-293e-4975-b4fd-9d259033f74c.png)


**核心思想$3$：模块中的滤波器组被扩展（即变得更宽而不是更深）**

具体操作见图$5$所示，作者将$Inception$模块中的$3$×$3$卷积核分解成并行的$3$×$1$和$1$×$3$的卷积，用来平衡网络的宽度和深度。

![图5](https://files.mdnice.com/user/15207/5336a865-81d3-4147-8309-d32c88270fe9.png)

**核心思想$4$：引入辅助分类器，加速深度网络收敛**

作者在文中提出辅助分类器在网络训练的早期不起作用，在网络训练的末期可以帮助获取更高的准确率。并且辅助分类分支起着正则化的作用，如果在侧分支使用$BN$层，那么网络的主分类器的性能会更好。但是只是单纯地使用$BN$层获得的增益还不明显，还需要一些相应的调整：增大学习速率并加快学习衰减速度以适用$BN$规范化后的数据；去除$Dropout$并减轻$L2$正则（因$BN$已起到正则化的作用）；去除$LRN$；更彻底地对训练样本进行$shuffle$；减少数据增强过程中对数据的光学畸变（因为$BN$训练更快，每个样本被训练的次数更少，因此更真实的样本对训练更有帮助）。在使用了这些措施后，$Inception$ $V2$在训练达到$Inception$ $V1$的准确率时快了$14$倍，并且模型在收敛时的准确率上限更高。如下图$6$所示，在侧分枝中使用了$BN$层。

![图$6$](https://files.mdnice.com/user/15207/641cdb80-8e6f-4f60-803b-924810b8a833.png)

**核心思想$5$：避免使用瓶颈层**

作者提出使用传统方法搭建卷积神经网络时候会使用一些池化操作以减小特征图大小。如图$7$左边先使用池化再进行卷积会导致瓶颈结构，违背$Inception$ $V2$的设计初衷；图$7$右边先使用卷积再进行池化，会增加计算成本：

![图$7$](https://files.mdnice.com/user/15207/96ab1724-0ef8-4570-95d1-9ba51acc18c8.png)


**核心思想$6$：减小特征图的尺寸**

作者采用了一种并行的降维结构，在加宽卷积核组的同时减小特征图大小，既减少计算量又避免了瓶颈结构。如下图$8$所示：

![图$8$](https://files.mdnice.com/user/15207/b3bab6d7-df33-4dbc-8bac-ef10cacf23e7.png)

最后我们来看一下在使用了以上的核心思想的$Inception$ $V3$网络整体结构，如下图$9$所示，其中$35$×$35$×$288$、$17$×$17$×$768$和$8$×$8$×$1280$三种输入的$Inception$结构分别对应上文中提到的图$2$、图$4$、图$5$。

![图$9$](https://files.mdnice.com/user/15207/67137807-39b5-450b-8fca-6526578438cb.png)

---
**核心思想$7$：Label Smoothing**

主要是为了消除训练过程中$label$-$dropout$的边缘效应。

对于每一个训练$example$ $x$, 模型计算每个label $k \in\{1 \ldots K\}$ 的概率： $p(k \mid x)=\frac{\exp \left(z_{k}\right)}{\sum_{i=1}^{K} \exp \left(z_{i}\right)}$, 其中 $z_{i}$ 是logits或未归一化的对数概 率。
训练集上单个example标签的实际概率分布（ground-truth distribution) 经过归一化后： $\sum_{k} q(k \mid x)=1$ 。为了简洁，我们忽略 $p$ 和 $q$ 对 $x$ 的依赖。我们定义单个example上的cross entropy为 $l=-\sum_{k=1}^{K} \log (p(k)) q(k)$ 。最小化cross entropy等价于最大化一个标签的对 数极大似然值的期望（expected log-likelihood of a label) , 这里标签是根据 $q(k)$ 选择的。cross entropy损失函数关于logits $z_{k}$ 是处处可 微的, 因此可以使用梯度下降来训练深度网络。其梯度有一个相当简单的形式： $\frac{\partial l}{\partial z_{k}}=p(k)-q(k)$, 它的范围是在-1 1之间。
对于一个真实的标签 $y:$ 对于所有的 $k=y$ 的情况，要 $q(y)=1$ 并且 $q(y)=1_{\circ}$ 在这种情况下，最小化交叉嫡等价于最大化正确标签 的对数似然。对于一个标签为 $y$ 的example $x$, 最大化 $q(k)=\delta_{k, y}$ 时的对数似然，这里 $\delta_{k, y}$ 是狄拉克 $\delta$ 函数。在 $k=y$ 时，狄拉克函数王 于1, 其余等于0。通过训练，正确logit的 $z_{y}$ 应该远大于其它 $z_{k}(z /=y), z_{y}$ 越大越好，但大是一个无终点的事情。这能够导致两个问
题, 1.导致过拟合：如果模型学习去将所有的概率分配到真实标签的逻辑单元上，泛化是没有保证的。2.它鼓励最大logit和其它logit的差 异 (KL距离) 越大越好，结合有界梯度（dounded gradient) $\frac{\partial l}{\partial z_{k}}$, 这降低了模型的适应能力。直觉上，适应能力的降低的原因应该是模 型对它的预测太过于自信。
作者提出了一个鼓励模型不过于自信的机制。如果目标是最大化训练标签的对数似然，那么这可能不是我们想要的，它对模型进行了 正则并且使得模型的适应性更强。该方法是非常简单的。考虑一个独立于训练example $x$ 的标签分布 $u(k)$, 和一个smoothing参数 $\epsilon_{\circ}$ 对于 一个训练标签为 $y$ 的example, 我们替代标签分布 $q(k \mid x)=\delta_{k, y}$ 为
$$
q^{\prime}(k \mid x)=(1-\epsilon) \delta_{k, y}+\epsilon u(k)
$$
新的分布是原始标签分布和一个指定分布 $u(k)$ 的混合，两部分的权重为 $1-6$ 和 $\epsilon_{\text {。 }}$ 这可以看做标签 $k$ 的分布是通过如下方式获得的：首 先, set it to the ground-truth lable $k=y ;$ 然后, 以权重 $\epsilon$, replace $k$ with a sample drown from the distribution $u(k)$ 。作者建议去使用 标签的先验分布作为 $u(k)$ 。在我们的实验中，我们使用了均匀分布 $(u(k)=1 / K)$
$$
q^{\prime}(k)=(1-\epsilon) \delta_{k, y}+\frac{\epsilon}{K}
$$
我们将这种改变$ground$-$truth$ $label$分布的方法称为$label$ $-$ $smoothing$ $regulariation或者$ $L S R_{\text {。 }}$
注意$LSR$达到了期望的目标：阻止最大的$logit$远大于其它$logit$。事实上，如果这发生了，则 $q(k)$ 将接近1, 而其它将接近0。这将导致一个很大的$cross-entropy$ $with$ $q^{\prime}(k)$, 因为, 不同于 $q(k)=\delta_{k, y}$, 所有的 $q^{\prime}(k)$ 有一个正的下界 ($positive$ $lower$ $bound$)
$LSR$的另一种损失函数可以通过研究交叉嫡损失函数来获得
$$
H\left(q^{\prime}, p\right)=-\sum_{k=1}^{K} \log p(k) q^{\prime}(k)=(1-\epsilon) H(q, p)+\epsilon H(u, p)
$$
因此，$ LSR$等价于用一对损失函数 $H(q, p)$ 和 $H(u, p)$ 来代替单个损失函数 $H(q, p)$ 。损失函数的第二项惩罚了预测标签的分布和先验 分布 $u$ 的偏差with相对权重 $\frac{\epsilon}{1-\epsilon}$ 注意, 因为 $H(u, p)=D_{K L}(u \| p)+H(u)$, 所以该偏差可以被KL散度捕获。当 $u$ 是均匀分布的时候, $H(u, p)$ 衡量的是预测分布 $p$ 和均匀分布之间的相似性，该相似性可以用负嫡 $-H(p)$ 来衡量, 但两者并不相等。作者没有对这一替代进行 研究。
在 ImageNet 中，因为有1000类, 所以作者令 $K=1000$ 。故 $u(k)=1 / 1000, \epsilon=0.1$ 。对于ILSVRC2012, 作者发现labelsmoothing regularization可以将top-1和top-5准确率提高$0.2\%$。



网络结构展示：

以下为$Inception$ $V3$使用到的$inception$模块的对应代码，分别是$InceptionA$—$InceptionE$。电脑上装好$Pytorch$和$torchvision$后，在本地$python$目录下的Lib\site-packages\torchvision\models中有详细的$Inception$ $V3$网络代码，有兴趣的小伙伴可以自行查看。


![$InceptionA$](https://files.mdnice.com/user/15207/634ed0aa-c35f-45ba-8247-3448503ce6de.png)
```python
# BasicConv2d是这里定义的基本结构：Conv2D-->BN，下同。
class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1) # 1

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
```



![$InceptionB$](https://files.mdnice.com/user/15207/f28657ca-811a-4066-a52d-dd2cd3b70936.png)


```python
class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
```



![$InceptionC$](https://files.mdnice.com/user/15207/5c89c3df-a2fa-465f-9e65-84cd1bcaff59.png)

```python
class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)
```


![$InceptionD$](https://files.mdnice.com/user/15207/d349ea77-008a-4f29-bb39-612806b74e48.png)

```python
class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)
```


![InceptionE](https://files.mdnice.com/user/15207/5661745c-fe7b-45bc-87b5-6fdaf658d264.png)
```python
class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
```


![辅助分类器$Utility of Auxiliary Classifiers$](https://files.mdnice.com/user/15207/85d746d3-1647-4f2d-9a69-010a96d38f34.png)


```python
class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x
```

### 总结

在$Inception$ $V2$和$V3$中，$V2$主要是在$V1$的基础上增加了$BN$层；$V3$提出了四种$Inception$模块，并引入辅助分类器，加速深度网络收敛。

另一方面，在$Inception$ $V3$中作者使用了$RMSProp$优化器和标签平滑，并在辅助分类器使用了$BatchNorm$，进一步加速网络收敛的速度。

论文的链接放在了下文的引用中，大家自行提取哈。

### 引用

- https://arxiv.org/abs/1512.00567
- https://zhuanlan.zhihu.com/p/30172532
- https://my.oschina.net/u/4597666/blog/4525757
- https://www.cnblogs.com/shouhuxianjian/p/7786760.html
- https://blog.csdn.net/weixin_44474718/article/details/99081062?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242
- https://zhuxiaoxia.blog.csdn.net/article/details/79632721?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-10.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-10.control



