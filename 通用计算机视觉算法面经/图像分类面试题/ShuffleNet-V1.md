## $ShuffleNet$ 系列算法之 $V1$

### 简介

$ShuffleNet$ $V1$ 是 $Face++$ 于 $2017$ 年提出的轻量级深层神经网络。在 $ImageNet$ 竞赛和 $MS$ $COCO$ 竞赛中均表现了比其他移动端先进网络更优越的性能。

主要有以下两个亮点：

- 提出 $pointwise$ $group$ $convolution$ 来降低 $PW$ 卷积（也即是 1\*1 卷积）的计算复杂度。
- 提出 $channel$ $shuffle$ 来改善跨特征通道的信息流动。

### $PointWise$ $Group$ $Convolution$

先简单介绍一下 $PointWise$ $Group$ $Convolution$（组卷积）的概念，如下图 $1$ 所示，左边为常见的普通卷积运算，输出的每一个维度的特征都需要输入特征的每一个维度经过计算获得，但这样的计算量会比较大。因而在 $Alexnet$、$ResNeXt$ 等网络中采用了 $group$ $convolution$，在输入特征图的通道方向执行分组操作得到最后的输入特征，如图 $1$ 中左边所示。$PointWise$ $Group$ $Convolution$ 只是 $group$ $convolution$ 的一种特殊形式，特殊的地方在于它的卷积核的核大小为 $1$\*$1$。

![图 $1$ 左边为普通卷积  右边为组卷积](https://files.mdnice.com/user/15207/654f8187-3ddb-4856-b060-f91da3ed90bb.png)

**那么为什么 $ShuffleNet$ 网络要使用逐点组卷积呢？**

其实是由于 $group$ $convolution$ 本身的问题导致的，我们知道使用 $group$ $convolution$ 的网络有很多，如 $Xception$，$MobileNet$，$ResNeXt$ 等。

$Xception$ 等模型采用了 $depthwise$ $convolution$，这是一种比较特殊的 $group$ $convolution$，此时分组数恰好等于通道数，意味着每个组只有一个特征图。这些网络存在一个很大的弊端是采用了密集的 $1$x$1$ 的 $pointwise$ $convolution$，在 $ResNeXt$ 模型中 $1$x$1$ $pointwise$ $convolution$ 基本上占据了 $93.4$%的乘加运算。那么在 $ShuffleNet$ 自然而然就提出了 $pointwise$ $group$ $convolution$ 来降低网络中 $1$×$1$ $pointwise$ $convolution$ 的计算量。

在原论文中作者也给出了 $ResNet$、$ResNeXt$ 和 $ShuffleNet$ 这三种模型中的一个 $bottleneck$ 的计算量对比，从下面的计算公式可知 $ShuffleNet$ 的计算量确实是最小的。下图 $2$ 为 $ResNet$ 的一个残差块，图 $3$ 分别为 $ResNeXt$（将图中 $3$×$3$ 的 $DWConv$ 换成 $3$×$3$ 的 $GConv$）和 $ShuffleNet$ 的 $bottleneck$。

$$
ResNet:   hw(1×1×c×m) + hw(3×3×m×m) + hw(1×1×c×m) = hw(2cm+9m^2)
$$

$$
ResNeXt:  hw(1×1×c×m) + hw(3×3×m×m)/g + hw(1×1×c×m) = hw(2cm+9m^2/g)
$$

$$
ShuffleNet:  hw(1×1×c×m)/g + hw(3×3×m) + hw(1×1×c×m)/g = hw(2cm/g+9m)
$$

![图 $2$ $ResNet$ 的残差块](https://files.mdnice.com/user/15207/950096f8-ec05-45bf-b8d8-56e5be13d34e.png)

![图 $3$ 左为 $ResNeXt$ 右为 $ShuffleNet$](https://files.mdnice.com/user/15207/0759acc4-4baf-4063-a4c5-e89b6de6e1be.png)

由上可知，虽然 $pointwise$ $group$ $convolution$ 可以降低计算量，但是如果多个组卷积堆叠在一起，会产生一个副作用：某个通道的输出结果，仅来自于一小部分输入通道，这个副作用会导致在组与组之间信息流动的阻塞，以及表达能力的弱化。那么我们如何解决这个问题呢？这用到了本文的第二个创新点—**$channel$ $shuffle$**。

### $channel$ $shuffle$

为达到特征之间通信目的，作者提出了 $channel$ $shuffle$。如图 $4$-$a$ 为正常采用组卷积提取出来的特征（相同颜色的通道表示是在同一个 $Group$）。图 $4$-$b$ 就是采用 $channel$ $shuffle$ 思想对 $group$ $convolution$ 之后的特征图进行“重组”，这样可以保证接下了采用的 $group$ $convolution$ 其输入来自不同的组，因此信息可以在不同组之间流转。图 $4$-$c$ 进一步的展示了这一过程并随机，其实是“均匀地打乱”。在程序上实现 $channel$ $shuffle$ 是非常容易的：假定将输入层分为 $g$ 组，总通道数为 $g$×$n$，首先将通道那个维度拆分为 $(g,n)$ 两个维度，然后将这两个维度转置变成 $(n,g)$，最后重新 $reshape$ 成一个维度 $g$×$n$。仅需要简单的维度操作和转置就可以实现均匀的 $shuffle$。采用 $channel$ $shuffle$ 之后就可以充分发挥 $group convolution$ 的优点，完美的避开其缺点啦。

![图 $4$](https://files.mdnice.com/user/15207/3ed06285-2c89-42de-8c9a-057d1f37c7f0.png)

$channel$ $shuffle$ 的 $Pytorch$ 代码如下：

```python
def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x
```

### $ShuffleNet$ 细节

#### 1、$ShuffleNet$ $unit$

$ShuffleNet$ 的基本单元是在一个残差单元的基础上采用上面的设计理念改进而成的。

![图 $5$](https://files.mdnice.com/user/15207/65e1e60d-7489-410d-9b66-da8aae6c49a8.png)

上图 $5$-$a$ 所示为一个包含 3 层的残差单元：首先是 $1$x$1$ 卷积，然后是 $3$x$3$ 的 $depthwise$ $convolution$，这里的 $3$x$3$ 卷积是瓶颈层（$bottleneck$），紧接着是 $1$x$1$ 卷积，最后是一个 $add$ 连接，将输入直接加到输出上。

现在，进行如下的改进：将密集的 $1$×$1$ 卷积替换成 $1$×$1$ 的 $group$ $convolution$，不过在第一个 $1$×$1$ 卷积之后增加了一个 $channel$ $shuffle$ 操作。另外 $3$×$3$ 的 $depthwise$ $convolution$ 之后没有使用 ReLU 激活函数。需要注意的是，**这里的 $stride$=$1$**。改进之后如图 $5$-$b$ 所示。

对于残差单元，如果 $stride$=$1$ 时，此时输入与输出 $shape$ 一致可以直接相加，而当 $stride$=$2$ 时，通道数增加，而特征图大小减小，此时输入与输出不匹配。为了解决这个问题在 $ShuffleNet$ 中，对原输入采用 $stride$=$2$ 的 $3$x$3$ $avg$ $pool$，这样得到和输出一样大小的特征图，然后将得到特征图与输出进行连接（$concat$），而不是相加。这样做的目的主要是降低计算量与参数大小，如图 $5$-$c$ 所示。

$stride$=$1$ 和 $stride$=$2$ 下的 $ShuffleNet$ $unit$ 代码如下：

```python
# stride$=1
class ShuffleNetUnitA(nn.Module):
    """ShuffleNet unit for stride=1"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                        1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=1,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.relu(x + out)
        return out

# -----------------------------------------------------------
# stride$=1
class ShuffleNetUnitB(nn.Module):
    """ShuffleNet unit for stride=2"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitB, self).__init__()
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=2,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        out = F.relu(torch.cat([x, out], dim=1))
        return out
        
```

#### 2、$ShuffleNet$ 整体网络结构

![图 $6$ $ShuffleNet$ 整体网络结构](https://files.mdnice.com/user/15207/5cecd1a1-f1c3-42a6-ad64-935e15aa17ca.png)

$ShuffleNet$ 的整体网络结构如图 $6$ 所示。首先是普通的 $3$x$3$ 的卷积和 $max$ $pool$ 。然后是三个 $stage$，每个 $stage$ 都是重复堆积若干 $ShuffleNet$ $unit$。对于每个 $stage$，第一个 $ShuffleNet$ $unit$ 的 $stride$=$2$，这样特征图 $width$ 和 $height$ 各降低一半，而通道数增加一倍。

后面的 $ShuffleNet$ $unit$ 都是 $stride$=$1$，特征图和通道数都保持不变。其中 $g$ 控制了 $group$ $convolution$ 中的分组数，分组越多，在相同计算资源下，可以使用更多的通道数，所以 $g$ 越大时，采用了更多的卷积核。当完成三个 $stage$ 后，采用 $global$ $pool$ 将特征图大小降为 $1$×$1$，最后是输出类别预测值的全连接层。

#### 3、实验结果

作者做了大量的对比实验来证明 $ShuffleNet$ 的优秀性能，这里给出一部分实验结果。

图 $7$ 给出了采用不同的 $g$ 值的 $ShuffleNet$ $V1$ 在 $ImageNet$ 上的表现结果。可以看到基本上当 $g$ 越大时，错误率越低，这是因为采用更多的分组后，在相同的计算约束下可以使用更多的通道数，或者说特征图数量增加，网络的特征提取能力增强，网络性能得到提升。注意 $Shuffle$ $1×$ 是基准模型，而 $0.5$× 和 $0.25$× 表示的是在基准模型上将通道数缩小为原来的 $0.5$ 和 $0.25$。

![图 $7$ 不同的 $g$ 值的 $ShuffleNet$ 的表现结果](https://files.mdnice.com/user/15207/2f93e6c9-5c3e-4af3-92f8-6fdd354ab503.png)

除此之外，作者还对比了不采用 $channle$ $shuffle$ 和采用之后的网络性能对比，如下图 $8$ 所示。可以清楚的看到，采用 $channle$ $shuffle$ 之后，网络性能更好，从而证明 $channle$ $shuffle$ 的有效性。

![图 $8$ 不采用 $channle$ $shuffle$ 和采用之后的网络性能对比](https://files.mdnice.com/user/15207/f4114766-7f3c-4c43-806f-9e283c6eb836.png)

作者也对比了 $ShuffleNet$ 与 $MobileNet$ 的计算量和精度，如下图 $9$ 所示。可以看到 $ShuffleNet$ 不仅计算复杂度更低，而且精度更好。

![图 $9$ $ShuffleNet$ 与 $MobileNet$ 的计算量和精度对比](https://files.mdnice.com/user/15207/a2d2296b-b738-4c16-b1b4-1cd95a156f3a.png)

其他一些实验对比结果大家可以阅读原论文获取。原论文链接放在了引用中，大家自提哈。

### 总结

$ShuffleNet$ 针对现大多数模型采用的逐点卷积存在的问题，提出了 $pointwise$ $group$ $convolution$ 和 $channel$ $shuffle$ 的处理方法，并基于这两个操作提出了一个 $ShuffleNet$ $unit$，最后在多个竞赛中证明了这个网络的效率和精度。

### 引用

- https://arxiv.org/abs/1707.01083
- https://zhuanlan.zhihu.com/p/32304419
- https://www.jianshu.com/p/c5db1f98353f
- https://www.cnblogs.com/hellcat/p/10318630.html
- https://blog.csdn.net/weixin_43624538/article/details/86155936
- https://www.sogou.com/link?url=hedJjaC291MBtMZVirtXo7CqjI0tE6P91nAMx5j2isv6gED9wOCRuw..
