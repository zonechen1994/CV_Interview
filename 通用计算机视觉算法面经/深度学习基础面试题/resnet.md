[阅读原文](https://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485417&idx=1&sn=e459bb2d9792e699d369dbc4e36370af&chksm=c241e4a5f5366db3330484db4a99a0c6ddf977676a21b34765d6caad55552f3ea42039b353c7&scene=178&cur_album_id=1860258784426672132#rd)


# 具体阐述一下ResNet网络的细节，你知道的ResNet网络的相关变种有哪些？



## 1. ResNet解决了什么问题？



首先在了解`ResNet`之前，我们需要知道目前`CNN`训练存在两大问题：



1. 梯度消失与梯度爆炸：因为很深的网络，选择了不合适的激活函数，在很深的网络中进行梯度反传，梯度在链式法则中就会变成0或者无穷大，导致系统不能收敛。然而梯度弥散/爆炸在很大程度上被合适的激活函数(**ReLU**)、流弊的网络初始化(**Kaiming初始化**`、`**BN**等Tricks)处理了。
2. 梯度弥散：当深度开始增加的时候, **accuracy**经常会达到饱和，然后开始下降，但这并不是由于过拟合引起的。如**56-layer**的**error**大于**20-layer**的**error**。



![image](https://cdn.nlark.com/yuque/0/2021/png/1160322/1617289648213-632ab81a-1f6f-4e78-a218-52a1513ae6dd.png)



**ResNet本身是一种拟合残差的结果，让网络学习任务更简单，可以有效地解决梯度弥散的问题。**

**ResNet**网络变种包括**ResNet V1**、**ResNet V2**、**ResNext**`以及**Res2Net**网络等。



## 2. ResNet网络结构与其性能优异的原因

ResNet残差块的结构如图所示。



![image](https://cdn.nlark.com/yuque/0/2021/png/1160322/1617289648432-57691f7b-dd0e-4442-a755-5a76add9a582.png)



`ResNet`网络的优点有：



- 引入跳跃连接，允许数据直接流向任何后续项。
- 引入残差网络，可以使得网络层数非常深，可以达到`1000`层以上。



同样，`ResNet`网络的设计技巧有：



- 理论上较深的模型不应该比和它对应的较浅的模型更差，较深的模型可以理解为是先构建较浅的模型，然后添加很多恒等映射的网络层。
- 实际上我们较深的模型后面添加的不是恒等映射，是一些非线性层，所有网络退化问题可以看成是通过多个非线性层来近似恒等映射是困难的。解决网络退化问题的方法就是让网络学习残差。



通过分析`ResNet`网络可以知道，`ResNet`可以被看做许多路径的集合，通过研究`ResNet`的梯度流表明，网络训练期间只有短路径才会产生梯度流，深的路径不是必须的，通过破坏性试验可以知道，路径之间是相互依赖的，这些路径类似集成模型，其预测准确率平滑地与有效路径的数量有关。



由于`ResNet`网络中存在很多`short cut`，所以`ResNet`又可以被视为很多路径的集合网络。相关实验表明，在`ResNet`网络训练期间，只有短路径才会产生梯度流动，说明深的路径不是必须的。通过破坏网络中某些`short cut`实验可以看出，在随机破坏了`ResNet`网络中的某些`short cut`后，网络依然可以训练，说明在网络中，即使这些路径是共同训练的，它们也是相互独立，不相互依赖的，可以将这些路径理解为集成模型，这也是理解`ResNet`网络的性能较好的一个方向。



## 3. ResNetv2的设计



首先，需要看下组对于主干以及分支网络的各种设计：



![image](https://cdn.nlark.com/yuque/0/2021/png/1160322/1617289648271-0dcb2cf9-9faf-4c8e-8f92-ee8045a9f4b1.png)



- (a) 原始$ResNet$模块，![img](https://g.yuque.com/gr/latex?f(z)%3DReLU(z))
- (b) 将$BN$移动到了![img](https://g.yuque.com/gr/latex?addition)
- (c) 将$ReLU$移动到![img](https://g.yuque.com/gr/latex?addition)。
- (d) 将$ReLU$移动在残差块之前，![img](https://g.yuque.com/gr/latex?f(z)%3Dz)
- (e) 将$BN$和$ReLU$移动到残差块之前，![img](https://g.yuque.com/gr/latex?f(z)%3Dz)
  在图中，$BN$会改变数据的分布，$ReLU$会改变值的大小，上面五个图都是work的，但是第五个图效果最好，具体效果如下：



![image-20210408214623963](/Users/zonechen/Library/Application Support/typora-user-images/image-20210408214623963.png)



具体效果为什么第五个好呢？先看下面的梯度求导吧！



## 4. ResNet的梯度公式推导

推导一下`ResNet`的前向与反向，步骤如下：



1. 首先回顾下`ResNet`的公式：



![img](https://g.yuque.com/gr/latex?y_%7Bl%7D%3Dh(x_%7Bl%7D)%2BF(x_%7Bl%7D%2C%20W_%7Bl%7D)%0A)

在这里，简化以上公式，令所有**identity**分支都是![img](https://g.yuque.com/gr/latex?h(x_%7Bl%7D)%3Dx_%7Bl%7D)，那么我们就可以得到：

![img](https://g.yuque.com/gr/latex?x_%7Bl%2B1%7D%3Dx_%7Bl%7D%2BF(x_%7Bl%7D%2C%20W_%7Bl%7D)%0A)

1. 这里我们就可以使用递归计算：



![F](https://g.yuque.com/gr/latex?x_%7Bl%2B2%7D%3Dx_%7Bl%2B1%7D%2BF(x_%7Bl%2B1%7D%2C%20W_%7Bl%2B1%7D)%3Dx_%7Bl%7D%2B%20F(x_%7Bl%7D%2C%20W_%7Bl%7D)%2BF(x_%7Bl%2B1%7D%2C%20W_%7Bl%2B1%7D)%0A)

则，不失一般性：

![img](https://g.yuque.com/gr/latex?x_%7BL%7D%3Dx_%7Bl%7D%2B%5Csum_%7Bi%3Dl%7D%5E%7BL-1%7DF(x_%7Bx_%7Bi%7D%2C%20W_%7Bi%7D%7D)%0A)



我们从上面公式可以看到深层$L$与浅层$l$之间的关系；则，假设损失函数为$loss$,那么反向传播公式为：

$\frac{\partial l o s s}{\partial x_{l}}=\frac{\partial \operatorname{los} s}{\partial x_{L}} \frac{\partial x_{L}}{\partial x_{l}}=\frac{\partial \operatorname{loss}}{\partial x_{L}}\left(1+\frac{\partial}{\partial x_{l}} \sum_{i=l}^{L-1} F\left(x_{i}, W_{i}\right)\right)$

而![img](https://g.yuque.com/gr/latex?%5Cfrac%7B%5Cpartial%20loss%7D%7B%5Cpartial%20x%7Bl%7D%7D)不会一直都是-1

从上式我们可以看到，`ResNet`有效得防止了：“当权重很小时，梯度消失的问题”。同时，上式中的优秀特点只有在假设![img](https://g.yuque.com/gr/latex?h(x_%7Bl%7D)%3Dx_%7Bl%7D)成立时才有效。所以，ResNet需要尽量保证亮点：

- 不要轻易改变identity分支的值。
- addition之后不再接受改变信息分布的层。

因此，在上面五组实验中，第五个(图e)效果最好的原因是：1) 反向传播基本符合假设，信息传递无阻碍；2）BN层作为pre-activation，起到了正则化的作用。

对于图(b),是因为$BN$在![img](https://g.yuque.com/gr/latex?addition)之后会改变分布，影响传递，出现训练初期误差下降缓慢的问题！

对于图(c),是因为这样做导致了$Residual$的分支的分布为负，影响了模型的表达能力。

对图(d),与图(a)在网络上相当于是等价的，指标也基本相同。



## 5. 其它相关解释

而除了像是从梯度反传的角度说明 <code>ResNet</code>比较好的解决了梯度弥散的问题，还有一些文章再探讨这些个问题。比如**“The Shattered Gradients Problem: If resnets are the answer, then what is the question?”**这篇工作中认为，即使BN过后梯度的模稳定在了正常范围内，但**梯度的相关性实际上是随着层数增加持续衰减的**。

而经过证明，ResNet可以有效减少这种相关性的衰减。对于 $L$ 层的网络来说，没有残差表示的Plain Net梯度相关性的衰减在 $\frac{1}{2^{L}}$ ，而ResNet的衰减却只有$\frac{1}{\sqrt{L}}$ 。这也验证了ResNet论文本身的观点，网络训练难度随着层数增长的速度不是线性，而至少是多项式等级的增长



## 6. 其他变种网络



### 6.1 WRNS(Wide Residual Networks)



在现有的经验上，网络的设计一般都是往深处设计，作者认为一味的增加深度不是一个有效的方法，`residual block`的宽度对网络的性能同样很有帮助，所以切入点在每一层的宽度上。



![image](https://cdn.nlark.com/yuque/0/2021/png/1160322/1617289651329-6c791427-46ac-4785-8cb8-439123d89564.png)



上图为论文中的图，原始的`ResNet`如图(a)与(b)所示，(b)是使用了`bottleneck`的`residual block`，而(c)与(d)是`WRN`的结构，主要是通道在原始`ResNet`通道数上加宽了`k`倍。

具体参数如下表：

![image](https://cdn.nlark.com/yuque/0/2021/png/1160322/1617289649713-5631744e-0a03-429d-a540-b297d87f6cb9.png)



而随着网络深度与宽度的加深，训练参数量过大会导致过拟合，作者提出在`residual block`里面加入`dropout`，也是就开头讲到的(d)。

从实验结果看，也能说明，当网络层数`depth`较浅，或者宽度 `k` 较小时，网络还不需要加`dropout`，但是当层数增加，宽度增加，参数量指数增大时，加入`dropout`可以有效防止`model`的`overfitting`。



### 6.2 ResNext



在`ResNet`提出`deeper`可以带来网络性质提高的同时，`WideResNet`则认为`Wider`也可以带来深度网络性能的改善。为了打破或`deeper`，或`wider`的常规思路，ResNeXt则认为可以引入一个新维度，称之为`cardinality`。



![image-20210408215128593](/Users/zonechen/Library/Application Support/typora-user-images/image-20210408215128593.png)

上图中左边为`ResNet`结构，右边为`cardinality=32`的`ResNeXt`结构（也就是含**32**个`group`）。其等效结构如下：

![image-20210408215204311](/Users/zonechen/Library/Application Support/typora-user-images/image-20210408215204311.png)



### 6.3 Res2Net



目前现有的特征提取方法大多都是用分层方式表示多尺度特征。分层方式即要么对每一层使用多个尺度的卷积核进行提特征（如目标检测中的`SPPNet`），要么就是对每一层提取特征进行融合（如`FPN`）。



本文提出的`Res2Net`在原有的残差单元结构中又增加了小的残差块，在更细粒度上，增加了每一层的感受野大小。`Res2Net`也可以嵌入到不同的特征提取网络中，如`ResNet, ResNeXt`等等。

在`Res2Net`中提出了一个新维度叫做`scale`。`Res2Net`的结构如下图所示：

![image](https://cdn.nlark.com/yuque/0/2021/png/1160322/1617289650204-807895c1-3da9-4bb7-bcab-9640a04d40b4.png)



上图左边是最基本的卷积模块。右图是针对中间的3x3卷积进行的改进。

首先对经过**1x1**输出后的特征图按通道数均分为$s$)（图中![img](https://g.yuque.com/gr/latex?s%3D4)

每一个$x_{i}$都会具有相应的**3x3**卷积，由![img](https://g.yuque.com/gr/latex?K_%7Bi%7D())的输出。



特征子集$x_{i}$与![img](https://g.yuque.com/gr/latex?K_%7Bi-1%7D())卷积，这样也可以看做是对特征的重复利用。

![image](https://cdn.nlark.com/yuque/0/2021/png/1160322/1617289650473-fd625fba-d58c-4e07-93cf-24a398adc57f.png?x-oss-process=image%2Fresize%2Cw_1500)



当然，`Res2Net`可以与像`SE`模块进行复用。



![image](https://cdn.nlark.com/yuque/0/2021/png/1160322/1617289650250-058ef985-4cbb-4e38-a0e2-cc977b83908f.png)



最后，论文中也给出了关于`depth`,`cardinality`以及`scale`的效果对比，如下图所示：



![image](https://cdn.nlark.com/yuque/0/2021/png/1160322/1617289650642-898e8f79-1075-4b42-9bcb-5370053fbc7b.png)

### 6.4 **ReXNet**

作者首先提出了，在传统网络的设计的中可能会存在Representational Bottleneck问题，并且该问题会导致模型性能的降低。其次，通过数学和实验研究探讨网络中出现的Representational Bottleneck问题。

论文中，先给定一个深度为L层的网络，通过$d_{0}$维的输入$X_{0} \in R^{d_{0} \times N}$可以得到个被编码为$X_{L}=\sigma\left(W_{L}\left(\ldots F_{1}\left(W_{1} X_{0}\right)\right)\right)$的特征，其中$W_{i} \in R^{d_{i} \times d_{i-1}}$为权重。

这里称$d_{i}>d_{i-1}$的层为expand层，称$d_{i}<d_{i-1}$的层为层。$f_{i}(\cdot)$为第i个点出的非线性函数，比如带有BN层的ReLU层，每个$f_{i}(\cdot)$表示第i个点非线性，如带有批归一化(BN)层的ReLU，$\sigma(\cdot)$为Softmax函数。

当训练模型的时候，每一次反向传播都会通过输入得到的输出与Label矩阵($T \in R^{d_{L} \times N}$)之间的Gap来进行权重更新。

因此，这便意味着Gap的大小可能会直接影响特征的编码效果。这里对CNN的公式做略微的改动为$X_{L}=\sigma\left(W_{L} *\left(\ldots F_{1}\left(W_{1} * X_{0}\right)\right)\right)$；式中$ *$和$W_{i}$分别为卷积运算和第i个卷积层核的权值。用传统的$W_{i} \hat{X}_{i-1}$重新排序来重写每个卷积，其中$W_{i} \in R^{d_{i} \times k_{i}^{2} d_{i-1}}$和$\hat{X}_{i-1} \in R^{k_{i}^{2} d_{i-1} \times w h N}$重新排序的特征，这里将第个特征写成:

​                      															  $\mathbf{X}_{i}=\left\{\begin{array}{ll}
f_{i}\left(\mathbf{W}_{i} \hat{\mathbf{X}}_{i-1}\right) & 1 \leq i<L \\
\sigma\left(\mathbf{W}_{L} \hat{\mathbf{X}}_{L-1}\right) & i=L
\end{array}\right.$

作者抛出了两个问题：

- **Softmax Bottleneck**

  由上面的卷积公式可以得知，交叉熵损失的输出为$\log X_{L}=\log \sigma\left(W_{L} X_{L-1}\right)$，其秩以$W_{L} X_{L-1}$的秩为界，即$\min \left(d_{L}, d_{L-1}\right)$。由于输入维度$d_{L-1}$小于输出维度$d_{L}$，编码后的特征由于秩不足而不能完全表征所有类别。这解释了Softmax层的一个Softmax bottleneck实例。能否通过引入非线性函数来缓解Softmax层的秩不足，性能得到了很大的改善？

-  **Representational bottleneck**

   作者推测，扩展channel大小的层(即层)，如下采样块，将有秩不足，并可能有Representational bottleneck。

   能否通过扩大权重矩阵的$W_{i}$秩来缓解中间层的Representational bottleneck问题？

   给定某一层生成的第$i$个特征，$X_{i}=f_{i}\left(W_{i} X_{i-1}\right) \in R^{d_{i} \times w h N}$，$\operatorname{rank}\left(X_{i}\right)$ 的阈值为$\min \left(d_{i}, d_{i-1}\right)$(假设$N>>d_{i}$)。这里$f(X)=X \circ g(X)$，其中$\circ$表示与另一个函数$g$的点乘。在满足不等式$\operatorname{rank}(f(X)) \leq \operatorname{rank}(X) \cdot \operatorname{rank}(g(X))$的条件下，特征的秩范围为：

   ​																				$$\operatorname{rank}\left(\mathbf{X}_{i}\right) \leq \operatorname{rank}\left(\mathbf{W}_{i} \mathbf{X}_{i-1}\right) \cdot \operatorname{rank}\left(g_{i}\left(\mathbf{W}_{i} \mathbf{X}_{i-1}\right)\right)$$

   因此，可以得出结论，秩范围可以通过增加$\left(W_{i} X_{i-1}\right)$的秩和用适当的用具有更大秩的函数$g_{i}$来替换展开，如使用$swish$或$ELU$激活函数，这与前面提到的非线性的解决方法类似。

   当$d_{i}$固定时，如果将特征维数$d_{i-1}$调整到接近$d_{i}$，则上式可以使得秩可以无限接近到特征维数。对于一个由连续的1×1,3×3,1×1卷积组成的bottleneck块，通过考虑bottleneck块的输入和输出通道大小，用上式同样可以展开秩的范围。

**针对Layer-Level秩**

​	作者生成一组由单一层组成的随机网络$f(WX)$：其中$W \in R^{d_{\text {out }} \times d_{\text {in }}}, X \in R^{d_{i n} \times N}$，$d_{out}$随机采样，$d_{in}$则按比例进行调整,来判断**Layer-Level**秩。

<img src="https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgl5wG9R7uR2ZOsMGWtJOgqb2VevexqMf6wyAPbbNeoRtdbT2X4sjKbtRg95icAsHqHuok5IIsSicjDA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:50%;" />

​		特征归一化后的秩$\left(\operatorname{rank}(f(W X)) / d_{\text {out }}\right)$是由每个网络产生。为了研究$f$而广泛使用了非线性函数。对于每种标准化Channel大小，作者以通道比例$\left(d_{i n} / d_{o u t}\right)$在[0.1, .0]之间		和每个非线性进行10,000个网络的重复实验。图a和b中的标准化秩的展示图。

**针对通道配置**

​		随机生成具有expand层(即$d_{\text {out }}>d_{\text {in }}$)的L-depth网络，以及使用少量的condense层的设计原则使得$d_{\text {out }}=d_{\text {in }}$，这里使用少量的condense层是因为condense层直接降低了模型容量。在这里作者将expand层数从0改变为$L-1$，并随机生成网络。例如，一个expand层数为0的网络，所有层的通道大小都相同(除了stem层的通道大小)。作者对每个随机生成的10,000个网络重复实验，并对归一化秩求平均值。结果如图c和d所示。

<img src="https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgl5wG9R7uR2ZOsMGWtJOgqbIhsyDUQpUSia07VSRman8CId8Fh9EyanuRI09M7C3yGZpHaHibBWfvLg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:45%;" />

​	此外，作者测试了采样网络的实际性能，每个配置有不同数量的expand层，有5个bottleneck，stem通道大小为32。数据集用CIFAR100，在表1中给出了5个网络的平均准确率。

<img src="https://mmbiz.qpic.cn/mmbiz_png/5ooHoYt0tgl5wG9R7uR2ZOsMGWtJOgqbjoeRibAsubevLf293mGDUH44uA0DG1BulasUOJAGCA2vrVELCv39fcw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:45%;" />

因此，作者这里给出了扩展给定网络秩的设计原则:

1. **在一层上扩展输入信道大小**;
2. **找到一个合适的非线性映射**;
3. **一个网络应该设计多个expand层**。

**结论**：**表征瓶颈(representation bottleneck)将发生在这些扩展层和倒数第2层**

解决方案：

	- 中间层处理；扩大卷积层的输入通道大小，替换ReLU6s来细化每一层
	- 替换ReLU6s来细化每一层；作者扩大了倒数第2层的输入通道大小，并替换了ReLU6



### 补充

关于`ResNet`的变种网络，之后再进行变种的其实还有蛮多的，后面我们会说到提高感受野的时候，再进行扩展，如**unipose(魔改res2net)**等。还有一些如**HO-ResNet**，结合了数值微分方程相关的，各位感兴趣的可以看看，基本上面试上问到的可能性不大。

------

