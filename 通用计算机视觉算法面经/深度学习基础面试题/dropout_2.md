
[阅读原文](https://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247486168&idx=1&sn=ce1920cf5ff9a78d2f24c4ad65632b06&chksm=c241e994f5366082788b9fe906e93fec7e92da55a66aaab342d98eaa7c39ef599114dc0740b5&scene=178&cur_album_id=1860258784426672132#rd)


### Stochastic Depth
$Stochastic$ $Depth$是采取类似于$Dropout$的思路，在$ResNet$块上随机进行对模块的删除，进而提高对模型的泛化能力。

如图所示，为$Stochastic$ $Depth$的具体做法。
![](https://files.mdnice.com/user/6935/8551a816-dd33-4072-9d5e-d482a5810cc0.png)

用数学化的语言来表示下该过程就是：

若网络总共有 $L$ 个$block$，我们给每个$block$都加上了一个概率$p_{l}$ 。

在训练时：
根据$p_{l}$ 用一个$bernoulli$随机变量生成每个$block$的激活状态 $b_{l}$，最终把$ResNet$的$bottleneck$ $block$，从$H_{l}=\operatorname{ReL} U\left(f_{l}\left(H_{l-1}\right)+idtentity\left(H_{l-1}\right)\right)$调整成了$H_{l}=\operatorname{ReLU}\left(b_{l} f_{l}\left(H_{l-1}\right)+idtentity\left(H_{l-1}\right)\right)$。

其中，当$b_{l}=0$时，表明这个$block$未被激活，此时$H_{l}=\operatorname{ReL} U\left(identity\left(H_{l-1}\right)\right)$。特别地是。其中$p_{l}$是从$p_{0}=1$线性衰减到$p_{L}=0.5$，即$p_{l}=1-\frac{l}{L}\left(1-p_{L}\right)$。

在预测的时候：

$block$被定义为：
$H_{l}^{T e s t}=\operatorname{ReL} U\left(p_{l} f_{l}\left(H_{l-1}^{\text {Test }}\right)+identity\left(H_{l-1}^{\text {Test }}\right)\right)$。**相当于将$p_{l}$与该层的残差做了一个权值融合了。**


**个人觉得这样$Drop$有以下两个好处**：
- ，这种引入随机变量的设计有效的克服了过拟合使模型有了更好的泛化能力。这种$Drop$的方式，本质上一种模型融合的方案。由于训练时模型的深度随机，预测时模型的深度确定，事实上是在测试时把不同深度的模型融合了起来。
- 以往的$Dropout$或者$DropConnect$都主要是在全连接层进行，这里是对整个网络进行$Drop$的。

这里给出一个参考代码如下：
```python
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=(out_channels * 4), kernel_size=1),
            nn.BatchNorm2d((out_channels * 4)),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=(out_channels * 4), kernel_size=1, stride=stride),
            nn.BatchNorm2d((out_channels * 4))
        )

    def forward(self, x, active):      
        if self.training:
            if active == 1:
                print("active")
                identity = x
                identity = self.downsample(identity)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = x + identity
                x = self.relu(x)
                return(x)
            else:
                print("inactive")
                x = self.downsample(x)
                x = self.relu(x)
                return(x)
        else:
            identity = x
            identity = self.downsample(identity)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.prob * x + identity
            x = self.relu(x)
            return(x)
```

### Cutout
目前为主，丢的主要是权重，或者是丢的是神经元。这里开始，我们要丢的是是网络的输入，当然网络输入不仅仅可以丢，也可以添加噪声($Cutmix$等)，这个是后面要做的内容。当然，还有一些对于输入图像进行$Drop$的操作(如$random$ $erase$)，我这里先打个样，看下如何对输入图像进行丢弃。**后面补充下，其它丢弃输入图像的操作。**

**先看看$Cutout$的做法：**

图像上进行随机位置和一定大小的$patch$进行$0-mask$裁剪。一开始使用裁剪上采样等变换出复杂轮廓的$patch$，后来发现简单的固定像素$patch$就可以达到不错的效果，所以直接采用正方形$patch$。

通过$patch$的遮盖可以让网络学习到遮挡的特征。$Cutout$不仅能够让模型学习到如何辨别他们，同时还能更好地结合上下文从而关注一些局部次要的特征。

$Cutout$的效果图如下所示：

![](https://files.mdnice.com/user/6935/f2acb078-3eef-433e-bc21-91af30024664.png)

参考代码如下：
```python
import torch
import numpy as np
 
 
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length
 
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
 
        mask = np.ones((h, w), np.float32)
 
        for n in range(self.n_holes):
            y = np.random.randint(h)  # 返回随机数/数组(整数)
            x = np.random.randint(w)
 
            y1 = np.clip(y - self.length // 2, 0, h) #截取函数
            y2 = np.clip(y + self.length // 2, 0, h) #用于截取数组中小于或者大于某值的部分，
            x1 = np.clip(x - self.length // 2, 0, w) #并使得被截取的部分等于固定的值
            x2 = np.clip(x + self.length // 2, 0, w)
 
            mask[y1: y2, x1: x2] = 0.
 
        mask = torch.from_numpy(mask)   #数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
        mask = mask.expand_as(img)  #把一个tensor变成和函数括号内一样形状的tensor
        img = img * mask
        return img
```

$Cutout$有两个超参，不同的任务，可以自己调调实验下效果。


### DropBlock
首先直观的从图片中看下$DropBlock$的具体做法：
![](https://files.mdnice.com/user/6935/2a08b46d-1679-4afc-b1cd-6a5061474f5d.png)


其中(b)表示的是随机$Dropout$的效果，(c)为$Drop$掉相邻的一整片区域，即按$Spatial$块随机扔。

其论文中的算法伪代码如下：

![](https://files.mdnice.com/user/6935/7a902473-0094-4d39-a05d-31014a54307b.png)

其中这个$\gamma$的值，是依赖于$keep\_prob$的值的。其计算过程如下：
$\gamma = \frac{1-keep\_prob}{block\_size^{2}}\frac{feat\_size^{2}}{(feat\_size-block\_size+1)^{2}}$

$keep\_prob$可以解释为传统的$dropout$保留激活单元的概率， 则有效的区域为$(feat\_size - block\_size + 1)^{2}$ ,$feat\_size$ 为$feature$ $map$的$size$. 实际上$DropBlock$中的$dropblock$可能存在重叠的区域, 因此上述的公式仅仅只是一个估计. 实验中$keep\_prob$设置为0.75~0.95, 并以此计算$\gamma$的值。

给出一个参考的$Pytorch$版本的代码：
```python
#!/usr/bin/env python
# -*- coding:utf8 -*-
import torch
import torch.nn.functional as F
from torch import nn
 
 
class Drop(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(Drop, self).__init__()
 
        self.drop_prob = drop_prob
        self.block_size = block_size
 
    def forward(self, x):
        if self.drop_prob == 0:
            return x
        # 设置gamma,比gamma小的设置为1,大于gamma的为0,对应第五步
        # 这样计算可以得到丢弃的比率的随机点个数
        gamma = self.drop_prob / (self.block_size**2)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
 
        mask = mask.to(x.device)
 
        # compute block mask
        block_mask = self._compute_block_mask(mask)
        # apply block mask,为算法图的第六步
        out = x * block_mask[:, None, :, :]
        # Normalize the features,对应第七步
        out = out * block_mask.numel() / block_mask.sum()
        return out
 
    def _compute_block_mask(self, mask):
        # 取最大值,这样就能够取出一个block的块大小的1作为drop,当然需要翻转大小,使得1为0,0为1
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask
```



结合上一篇的三种$Drop$策略，我们主要从主要作用在全连接网络的$Dropout$，作用在$Channel$层面的$Spatial$ $Dropout$,作用在$Layer$层面的$Stochastic$ $Dropout$，作用在$Feature$ $map$层面的$DropBlock$，作用在输入层面的$Cutout$等方式。给大家梳理了各个$Drop$方案，后面有一些列的工作是针对输入提出的正则化技巧(数据增强)，在后面的文章，我们再进行补充～

这些方案具体怎么用？不好意思，需要你针对你自己的任务自己去调了。

在这里，我们要谈下，**为何BN提出后，Dropout就不用了呢？**  

### Dropout与BN不和谐共处


首先我们聊下在$Pytorch$中$BN$的$API$：
```python
nn.BatchNorm2d(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
```
- num_features:输入数据的通道数，归一化时需要的均值和方差是在每个通道中计算的
- eps: 滑动平均的参数，用来计算$running\_mean$和$running\_var$
- affine:是否进行仿射变换，即缩放操作
- track_running_stats:是否记录训练阶段的均值和方差，即running_mean和running_var

对于$BN$层的状态，包含了$5$个参数：
- weight:缩放操作的 $\gamma$。
- bias: 缩放操作的$\beta$
- running_mean: 训练阶段统计的均值，在测试的时候可以用到
- running_var: 训练阶段统计的方差，测试的阶段用
- num_batches_tracked，训练阶段的batch的数目，如果没有指定momentum，则用它来计算running_mean和running_var。一般momentum默认值为0.1，所以这个属性暂时没用。

假设我们的输入$tensor$的维度是$(4,3,2,2)$,那么我们我们在做$BN$的时候，我们在$channel$维度中“抽”出来一个通道的数据，则其维度为$(4,1,2,2)$。我们需要对这$16$个数据求均值$\mu$跟方差$\sigma$，并用求得的均值与方差归一化，再缩放数据，得到$BN$层的输出。

我们需要用滑动平均公式来更新$running\_mean$与$running\_var$，$momentum$默认为0.1.

$$
running\_mean = (1-momentum) * running\_mean + momentum * \mu 
$$

$$
running\_var = (1-momentum) * running\_var + momentum * \sigma 
$$


答：**Dropout在网络测试的时候神经元会产生“variance shift”，即“方差偏移”**。试想若有图一中的神经响应$X$，当网络从训练转为测试时，$Dropout$ 可以通过其随机失活保留率（即 $p$）来缩放响应，并在学习中改变神经元的方差，而 $BN$ 仍然维持 $X$ 的统计滑动方差($running\_var$)。这种方差不匹配可能导致数值不稳定。而随着网络越来越深，最终预测的数值偏差可能会累计，从而降低系统的性能。事实上，如果没有 $Dropout$，那么实际前馈中的神经元方差将与 $BN$ 所累计的滑动方差非常接近，这也保证了其较高的测试准确率。

下面有张图，也比较清楚的反映了，$Dropout$与$BN$在一起使用存在的问题：

![](https://files.mdnice.com/user/6935/50b2be99-2774-4e93-ac64-112d758e7873.png)


那么怎么解决这样的**variance shift**的问题呢？有两种方案：
- 在$BN$之后，连接一个$Dropout$。
- 修改 $Dropout$ 的公式让它对方差并不那么敏感。有工作是进一步拓展了高斯$Dropout$(即不是满足伯努利分布，而是Mask满足高斯分布)，提出了一个均匀分布$Dropout$，这样做带来了一个好处就是这个形式的$Dropout$（又称为$“Uout”$）对方差的偏移的敏感度降低了，总得来说就是整体方差偏地没有那么厉害了。而实验结果也是第二种整体上比第一个方案好，显得更加稳定。
