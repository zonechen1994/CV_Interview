## $Inception V4$ / $Inception-Resnet$

随着何凯明等人提出的$ResNet$结构后，等人在基于$Inception$ $v3$的基础上，引入了残差结构，提出了$Inception$-$resnet$-$v1$和$Inception$-$resnet$-$v2$，并修改$Inception$模块，提出了$Inception$ $v4$结构。并且作者在基于$Inception$ $v4$搭建的网络在不引入残差结构的基础上也能达到和$Inception$-$resnet$-$v2$结构相似的结果。下面就让我们一起来具体看一下这三个网络到底是什么样子的吧！

### $Inception V4$

在下面的所有网络结构图中，所有后面不带$V$的卷积，用的都是$same-padding$，也就是输出的特征图大小等于输入的特征图大小；带$V$的使用的是$valid-padding$，表示输出的特征图大小是会逐步减小的。

#### 网络结构

如下图$1$所示为$Inception$ $V4$网络结构，可以看出网络整体主要是由$Stem$、$Inception$和$Reduction$这三个模块组成的，其中$Inception$模块包含了$Inception-A$、$Inception-B$和$Inception-C$这三个子模块；$Reduction$模块包含了$Reduction$-$A$和$Reduction$-$B$这两个模块。

![图$1$ $Inception$ $V4$网络结构](https://files.mdnice.com/user/15207/6d2f3177-0806-4526-ae3d-3c97a0471b1f.png)

#### $Stem$结构
如下图2所示为$Inception$ $V4$的$Stem$结构模块，并且$Inception$ $V4$ 和$Inception-Resnet-V2$中都使用了这个结构。

![图$2$ $Inception$ $V4$的$Stem$结构](https://files.mdnice.com/user/15207/f9e49313-9c60-4635-bad0-069521c7c86d.png)

#### $Inception$ $V4$的$InceptionA-C$结构

图$3$、图$4$和图$5$分别为$Inception-A$、$Inception-B$和$Inception-C$的结构图，可以看出来这三个子模块和$InceptionV3$中的子模块是非常相似的，只是相比于$InceptionV3$中五个子模块来说少了两个。

![图$3$ $Inception$ $V4$的$Inception-A$结构](https://files.mdnice.com/user/15207/91a207ce-94d5-4dc2-9d56-9fe70c01ad94.png)


![图$4$ $Inception$ $V4$的$Inception-B$结构](https://files.mdnice.com/user/15207/0bda29d6-465c-4a53-bda3-63d6d9efb6a0.png)


![图$5$ $Inception$ $V4$的$Inception-C$结构](https://files.mdnice.com/user/15207/e13d28ec-ab1a-412f-9e7d-e4632dc4e07c.png)

#### $Inception$ $V4$的$ReductionA、B$结构
$Inception$ $v4$中引入了专用的$reduction$ $block$，它被用于改变网格的宽度和高度。$Reduction-A$实现了从$35$x$35$到$17$x$17$的尺寸缩减），$Reduction-B$（从$17$x$17$到$8$x$8$的尺寸缩减）。如下图6和图5所示，为$Reduction-A、B$结构。

![图$6$ $Inception$ $V4$的$Reduction-A$结构](https://files.mdnice.com/user/15207/3f15dae7-467d-4ef6-b0ae-f65f81ad2480.png)


![图$7$ $Inception$ $V4$的$Reduction-B$结构](https://files.mdnice.com/user/15207/fb038190-d20e-4c0d-ae39-1c419492c535.png)


### $Inception-Resnet$

作者受到$ResNet$的优越性能启发后，同时提出了$Inception$ $ResNet$,它有两个子版本：$v1$和$v2$。在我们介绍其结构之前，先看看这两个子版本之间的微小差异。

- $Inception-ResNet$ $v1$ 的计算成本和 $Inception$ $v3$ 的接近。
- $Inception-ResNet$ $v2$的计算成本和$Inception$ $v4$ 的接近。
- 它们有不同的$stem$，正如$Inception$ $v4$部分所展示的，$Inception$ $ResNet$ $V2$和$Inception$ $V4$使用同样的$Stem$。
- 两个子版本都有相同的$Inception$模块$A$、$B$、$C$和缩减块结构。唯一的不同在于超参数设置。

#### 网络结构

如下图$8$所示为$Inception-Resnet$的网络结构图。可以看出网络整体是由$Stem$、$Inception$和$Reduction$这三个模块组成的，其中$Inception$模块包含了$Inception$-$resnet$-$A$、$Inception$-$resnet$-$B$和$Inception$-$resnet$-$C$这三个子模块；$Reduction$模块包含了$Reduction$-$A$和$Reduction$-$B$这两个模块。

![图$8$ $Inception-Resnet$结构](https://files.mdnice.com/user/15207/4476bdf5-548b-4021-8276-a2d72d4a7aaf.png)

#### $Inception-Resnet$-$V1$的$Stem$结构
如下图$9$所示为$Inception-Resnet$-$V1$的$Stem$结构图，$Inception$ $ResNet$ $V2$和$Inception$ $V4$使用了如上图$2$中所示的同样的$Stem$结构。

![图$9$ $Inception-Resnet$-$V1$的$Stem$结构](https://files.mdnice.com/user/15207/a9a6476d-f25a-4472-b794-84b3416daec5.png)


#### $Inception$-$resnet$结构
如下图$10$-图$13$分别为$Inception$-$resnet$-$V1$的$Inception$-$resnet$三个子模块和$Reduction$-$B$模块；图$14$-图$17$分别为$Inception$-$resnet$-$V2$的$Inception$-$resnet$三个子模块和$Reduction$-$B$模块。$Inception$-$resnet$-$V1$和$Inception$-$resnet$-$V2$的$Reduction$-$B$模块和上图$6$的$Inception$ $V4$的 $Reduction$-$A$保持一致。

从这么多图中可以看出，$Inception$模块中的$A$、$B$、$C$中的池化层被残差连接所替代，并在残差加运算之前有额外的$1$x$1$卷积。

![图$10$ $Inception$-$resnet$-$V1$的$Inception$-$resnet$-$A$模块](https://files.mdnice.com/user/15207/d6d02bae-e1b7-45fd-a39e-08cb4ab14461.png)


![图$11$ $Inception$-$resnet$-$V1$的$Inception$-$resnet$-$B$模块](https://files.mdnice.com/user/15207/d2f37040-6244-4143-8ea6-605b5c116f75.png)


![图$12$ $Inception$-$resnet$-$V1$的$Inception$-$resnet$-$C$模块](https://files.mdnice.com/user/15207/a0b05c9e-037d-4563-aab6-d676064d1cb7.png)


![图$13$ $Inception$-$resnet$-$V1$的$Reduction$-$B$模块](https://files.mdnice.com/user/15207/a4115b03-3542-4fd8-a9f1-4d95f61e98e6.png)

---


![图$14$ $Inception$-$resnet$-$V2$的$Inception$-$resnet$-$A$模块](https://files.mdnice.com/user/15207/62fd9bda-7de6-48d2-b862-70b54d4414ab.png)


![图$15$ $Inception$-$resnet$-$V2$的$Inception$-$resnet$-$B$模块](https://files.mdnice.com/user/15207/de3e8cdc-90ec-4ab1-8294-20c1ca71e53d.png)


![图$16$ $Inception$-$resnet$-$V2$的$Inception$-$resnet$-$C$模块](https://files.mdnice.com/user/15207/ee28ae42-6ce9-4dd0-af74-5591255e1cdc.png)


![图$17$ $Inception$-$resnet$-$V2$的$Reduction$-$B$模块](https://files.mdnice.com/user/15207/149ddde1-30f8-4330-8b76-a8d5b95ff015.png)

#### 针对深网络结构设计的衰减因子

作者在训练过程中发现如果卷积核的数量超过$1000$，则网络架构更深层的残差单元将导致网络崩溃。即在迭代几万次之后，平均池化的前面一层就会生成很多的$0$值。即便通过调低学习率，增加$BN$层都没有任何改善。不过作者发现如果在将残差聚合之前，对残差进行缩放变小的话，可以让模型稳定训练，缩放值值通常选择$0.2$到$0.3$之间。如下图$18$所示。

![图$18$ 衰减因子](https://files.mdnice.com/user/15207/3d46ff6e-2d9d-4e82-8353-59ba0b1a59fd.png)

#### 训练结果对比
如下图$19$所示为作者对比了$Inception$-$resnet$、$Inception$-$V4$和$Inception$-$V3$的训练精度。从图中我们可以看出：
- 在$Inception$-$resnet$-$v1$与$Inception$ $v3$的对比中，$Inception$-$resnet$-$v1$虽然训练速度更快，不过最后结果有那么一丢丢的差于$Inception$ $v3$；
- 而在$Inception$-$resnet$-$v2$与$Inception$ $v4$的对比中，$Inception$-$resnet$-$v2$的训练速度更块，而且结果比$Inception$ $v4$也更好一点。所以最后胜出的就是$Inception$-$resnet$-$v2$。

![图$19$](https://files.mdnice.com/user/15207/83caf18a-e2d6-4220-9079-a53f99336b67.png)

### 总结
在本次分享的论文中，作者提出了$Inception$ $V4$、$Inception$-$resnet$-$V1$和$Inception$-$resnet$-$v2$这三种模型，总的来说改善了在$Inception$ $V3$中的$Inception$模块，并引入$Reduction$结构用来改变网络的高度和宽度。

另一方面也针对三种不同的网络设计了不同的$Stem$，并针对深网络在训练时容易出现网络崩溃的现象设计了衰减因子来保证网络训练的稳定。

且在$Inception$-$ResNet$结构中，只在传统层的上面使用$BN$层，而不在合并层上使用$BN$，解决了因为具有大量激活单元的层会占用过多的显存的问题。

论文的链接放在了下文的引用中，大家自行提取哈。

### 引用

- http://arxiv.org/abs/1602.07261
- https://my.oschina.net/u/4597666/blog/4525757
- https://www.cnblogs.com/shouhuxianjian/p/7786760.html
- https://blog.csdn.net/weixin_44474718/article/details/99081062?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242
- https://zhuxiaoxia.blog.csdn.net/article/details/79632721?utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromMachineLearnPai2~default-10.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromMachineLearnPai2~default-10.control