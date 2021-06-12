## $ShuffleNet$ 系列算法之 $V2$

### 简介

$ShuffleNet$-$V2$，它是由旷视提出的 $V1$ 升级版本，能够在同等复杂度下，比 $ShuffleNet$-$V1$ 和 $MobileNet$-$v2$ 更准确。

主要有以下一些亮点：

- 提出了四个高效网络设计指南 $G1$、$G2$、$G3$、$G4$
- 针对 $ShuffleNet$-$V1$ 的两种 $unit$ 做了升级

### 论文前期实验

在 $ShuffleNet$-$V2$ 这篇论文中，作者先是做了两个实验来抛砖引玉，为下文提出的四个高效网络设计指南以及创新点埋下伏笔。

#### 1、网络执行效率准则

目前衡量模型复杂度的一个通用指标是 $FLOPs$，具体指的是 $multiply-add$ 数量。在论文中作者一上来先提出了一个问题：

**是不是准确率和 $FLOPs$（浮点数计算的数量）就可以代表一个网络模型的性能呢？**

为此他们做了如下的实验：比较了分别以 $FLOPs$ 和 $Accuracy$ 为变量下不同平台下各个模型的推理时间。实验结果如下图 $1$ 所示。

![图 $1$](https://files.mdnice.com/user/15207/f6e561cd-18af-44e1-81bd-42acf405a247.png)

从图中我们可以看出：

$a$、在**不同平台下**同一个模型的推理速度是不同的；

$b$、**相同精度或相同 $FLOPs$** 下的模型推理时间是不同的。

作者经过分析后，认为出现这种情况的原因主要有：

- 对推理速度影响较大的因素，但没有影响到 $FLOPs$。例如：内存访问成本 $MAC$($memory$ $access$ $cost$)和并行度($degree$ $of$ $parallelism$)。

- 运行平台不同。不同的运行平台，针对卷积等操作有一定的优化，例如 $cudnn$ 等。

据此作者提出了 $2$ 个网络执行效率对比的准则：论文也是按照以下这两个准则，对多种网络（包括 $shufflenetv2$)进行评估。

**(1)使用直接度量方式如速度代替 FLOPs。**

**(2)在同一环境平台上进行评估。**

#### 2、模型推理耗时

如下图 $2$ 所示，作者分别在 $GPU$ 和 $CPU$ 上对 $ShuffleNet$-$V1$，$MobileNet$-$V2$ 的推理时间进行了测试。从图中可以看出，整个推理时间被分解用于不同的操作。处理器在运算的时候，不光只是进行卷积运算，也在进行其他的运算，特别是在 $GPU$ 上，卷积运算只占了运算时间的一半左右。作者将卷积部分认为是 $FLOPs$ 操作。虽然这部分消耗的时间最多，但其他操作包括数据 $IO$、$shuffle$ 和逐元素操作（$Elemwise$，$ReLU$ 等）也占用了相当多的时间。因此，再次确认了模型只使用 $FLOPs$ 指标对实际运行时间的估计是不够准确的。

![图 $2$ $ShuffleNet$-$V1$，$MobileNet$-$V2$ 在 $GPU$ 和 $CPU$ 上的推理时间占比](https://files.mdnice.com/user/15207/4af68bc3-70cf-4429-887e-a9ad521c98cb.png)

### 四个高效网络设计指南

综上两个实验，作者提出了四个高效网络设计指南 $G1$、$G2$、$G3$、$G4$。下面让我们一起来看看这四个东东都是什么吧。

#### $G1$:$Equal$ $channel$ $width$ $minimizes$ $memory$ $access$ $cost$

$G1$ 的意思为：卷积的输入输出具有相同 $channel$ 的时候，内存消耗是最小的。作者以 $1$×$1$$PW$ 卷积为例，假设输入和输出的特征通道数分别为 $c_1$ 和 $c_2$，特征图的大小为 $h$×$w$，则 $1$×$1$$PW$ 卷积的 $FLOPs$ 为:

$$
B=(hwc_1*1*1*c_2)=hwc_1c_2
$$

对应的 MAC($memory$ $access$ $cost$)为：

$$
MAC=hwc_1+hwc_2+1*1*c_1*c_2=hw(c_1+c_2)+c_1c_2
$$

并且我们知道以下均值不等式：

$$
c_1+c_2 ≥ 2\sqrt{c_1c_2}
$$

最后整理一下上面的式子可以知道：

$$
MAC≥2\sqrt{hwB}+\frac{B}{hw}
$$

那么在相同 $FLOPs$ 即当 $B$ 不变的时候，只有当 $c1$=$c2$ 的时候，$MAC$ 才能最小。如下图 $3$ 所示，作者也做了一系列的实验，从图中可看出，只有当 $c1$=$c2$ 的时候，在 $GPU$ 或者 $CPU$ 下的推理速度是最快的。

![图 $3$ 不同输入和输出下的推理速度](https://files.mdnice.com/user/15207/c256b8cd-d8c9-4c6a-a509-7dfe34850139.png)

---

#### $G2$：$Excessive$ $group$ $convolution$ $increases$ $MAC$

$G2$ 意思为：过多的分组卷积操作会增大 $MAC$，从而使模型速度变慢。和前面同理，$g$ 为分组的数量，带 $group$ 操作的 $1$×$1$ 卷积的 $FLOPs$ 为：

$$
B=hw(c_1*1*1*c_2)/g=hwc_1c_2/g
$$

其 $MAC$ 值为：

$$
MAC=hw(c_1+c_2)+\frac{c_1c_2}{g}=hwc_1+\frac{Gg}{c_1}+\frac{B}{hw}
$$

可以看出，在 $B$ 不变时，$g$ 越大，$MAC$ 也越大。如下图 $4$ 所示，作者也做了实验对比，从图可以看出随着 $g$ 越大，在 $GPU$ 和 $CPU$ 的推理速度也越慢。

![图 $4$ 不同 $g$ 值下 $GPU$ 和 $CPU$ 的推理速度](https://files.mdnice.com/user/15207/9a70fb64-9dc6-46ce-bd31-e5959218d945.png)

---

#### $G3$：$Network$ $fragmentation$ $reduces$ $degree$ $of$ $parallelism$

$G3$ 的意思是网络内部分支操作会降低并行度。作者认为，模型中的网络结构太复杂（分支和基本单元过多）会降低网络的并行程度，模型速度越慢。文章用了一个词：$fragment$，翻译过来就是分裂的意思，可以简单理解为网络的单元或者支路数量。

为了研究 $fragment$ 对模型速度的影响，作者做了第四个实验。如图 $5$ 所示，$1$-$fragment$-$series$ 表示单个卷积层；$2$-$fragment$-$series$ 表示一个 $block$ 中有 $2$ 个卷积层串行，也就是简单的叠加；$4$-$fragment$-$parallel$ 表示一个 $block$ 中有 $4$ 个卷积层并行，类似 $Inception$ 的整体设计。图 $6$ 为测试结果，可以看出在相同 $FLOPs$ 的情况下，单卷积层（$1$-$fragment$）的速度最快。

![图 $5$](https://files.mdnice.com/user/15207/2d005608-4227-4582-bc20-62605f16be45.png)

![图 $6$ 不同分支结构下 $GPU$ 和 $CPU$ 下的推理速度](https://files.mdnice.com/user/15207/0e3954e0-4d67-4b80-a9cd-e26baca697c5.png)

---

#### $G4$：$Element$-$wise$ $operations$ $are$ $non$-$negligible$

$G4$ 的意思为：$Element$-$wise$ 操作不能被忽略。如图 $2$ 所示，$IO$ 操作例如数据拷贝虽然 $FLOPs$ 非常低，但是带来的时间消耗还是非常明显的，尤其是在 $GPU$ 上。元素操作操作虽然基本上不增加 $FLOPs$，但是所带来的时间消耗占比却不可忽视。于是作者做了一个实验，采用的是 $Resnet50$ 的 bottleneck，除去跨层链接 $shortcut$ 和 $ReLU$ 之后测试其推理速度如下图 $7$ 所示，在 $GPU$ 和 $ARM$ 结构上都获得了接近 $20$% 的提速。

![图 $7$ $shortcut$ 和 $ReLU$ 的推理速度](https://files.mdnice.com/user/15207/3182354c-0d7a-43a3-8c4e-0a20d71377a9.png)

### $ShuffleNet$-$V2$ 设计

终于到了最后的部分了！这一部分，作者根据之前所提出的设计指南，在 $Shufflenet$-$V1$ 的基础上进行修改，得到了 $ShuffleNet$-$V2$。

#### 1、$ShuffleNet$-$V2$ 的 $unit$ 升级

作者在遵循 G1-G4 的设计准则的条件下，对于 $ShuffleNet$-$V1$ 的 $unit$ 进行了改进，如下图 $8$ 所示为
$ShuffleNet$-$V1$ 和 $V2$ 的 $unit$ 对比。图 $8$-$a$ 和图 $8$-$b$ 分别为 $ShuffleNet$-$V1$ 中 $stride$=$1$ 和 $stride$=$1$ 的 $unit$，图 $8$-$c$ 和图 $8$-$d$ 分别为对应的改进版。

![图 $8$](https://files.mdnice.com/user/15207/86e2e720-b3a1-41b9-ae87-123f1b41e0a7.png)

作者分析了原 $ShuffleNet$-$V1$ 中违背 $G1$-$G4$ 的一些结构：

1、图 $8$-$a$ 中的逐点组卷积增加了 $MAC$ 违背了 $G2$；

2、图 $8$-$a$ 和 $8$-$b$ 中过多的分组违背了 $G3$；

3、图 $8$-$a$ 和 $8$-$b$ 中卷积输入和输出不相等违背了 $G1$;

4、图 $8$-$a$ 中使用了 $add$ 操作违背了 $G4$;

针对于以上问题，作者在 $ShuffleNet$-$V2$ 中做了以下改进：

1、对于 $stride$=$1$ 的模块来说，如图 $8$-$c$，在每个单元的开始，通过 $Channel$ $split$ 将 $c$ 特征通道的输入被分为两支，分别为 $c$−$c'$ 和 $c'$ 个通道。按照准则 $G3$，一个分支的结构仍然保持不变，另一个分支由三个卷积组成，为了满足 $G1$，令输入和输出通道相同。与 $ShuffleNet$-$V1$ 不同的是，两个 $1$×$1$ 卷积不再是组卷积($GConv$)(遵循 $G2$)，因为 $Channel$ $Split$ 分割操作已经产生了两个组。

2、卷积之后，把两个分支拼接($Concat$)起来，避免使用 $Add$ 操作(遵循 $G4$)，从而通道数量保持不变 (遵循 $G1$)。然后进行与 $ShuffleNet$-$V1$ 相同的 $Channel$ $Shuﬄe$ 操作来保证两个分支间能进行信息交流(将 shuffle 移到了 concat 之后遵循 $G3$)。

3、对于 $stride$=$2$ 的模块来说，如图 $8$-$d$，使用一个 $3$×$3$ 的 DW 卷积和 $1$×$1$ 的卷积来替代 $V1$ 当中的平均池化操作。也将原来的 $1$×$1$ 组卷积换成了普通的 $1$×$1$ 卷积(遵循 $G2$)。

如下所示为 $pytorch$ 版本 $ShuffleNet$-$V2$ 的结构单元代码：

```python
def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out
```

#### 2、$ShuffleNet$-$V2$ 的整体网络结构

![图 $9$ $ShuffleNet$-$V2$ 的整体网络结构](https://files.mdnice.com/user/15207/220e6ce8-8762-4af5-b4a4-fe0a4df0afe0.png)

$ShuffleNet$-$V2$ 的整体网络结构如图 9 所示。基本上延续了和 $V1$ 相似的结构，每个 $stage$ 都包含了若干 $stride$=$1$ 和 $stride$=$2$ 的模块，需要主要的是，每经过一个 $stage$ 通道数都会增加一倍的原因是由于 $stride$=$2$ 的模块(图 $8$-$d$)两个分支的通道数都等于输入通道，再经过 $concat$ 之后输出通道自然增加两倍。

#### 2、$ShuffleNet$-$V2$ 实验结果

![图 $9$ $ShuffleNet$-$V2$ 的实验结果](https://files.mdnice.com/user/15207/24c3ec37-7be6-4649-8442-32d82d6688f3.png)

如图 $9$ 所示为 $ShuffleNet$-$V2$ 的实验结果，作者比较了 $ShuffleNet$-$V2$ 和 $MobileNet$ 系列、$DenseNet$、$Xception$ 等网络在 $GPU$ 和 $CPU$ 下的推理速度和精度。从图中可知，$ShuffleNet$-$V2$ 在精度和推理速度上都是更加杰出的。

### 总结

1、$ShuffleNet$-$V2$ 完善了网络性能对比的准则，以 $FLOPs$、推理速度和平台这三个因素来综合评价网络的性能。

2、提出了四个高效网络设计指南，并据此设计了 $ShuffleNet$-$V2$ 中的结构单元。

### 引用

- https://arxiv.org/pdf/1807.11164.pdf
- https://www.jianshu.com/p/c5db1f98353f
- https://zhuanlan.zhihu.com/p/67009992
- https://zhuanlan.zhihu.com/p/48261931
- https://blog.csdn.net/weixin_44474718/article/details/91041343
- https://blog.csdn.net/weixin_43624538/article/details/86155936
