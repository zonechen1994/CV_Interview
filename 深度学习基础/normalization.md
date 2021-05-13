


# Normalization汇总！！！

[阅读原文](https://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485824&idx=1&sn=49aa89fd0e866d24e9923a5d7d5dce69&chksm=c241eaccf53663daa6d07d7ee9763b5c4d5ae17c562f4b35d625be0b0328b8cab12844b1fbee&scene=178&cur_album_id=1860258784426672132#rd)

## 1. 对数据部分做归一化

**主要且常用**的归一化操作有**BN，LN，IN，GN**，示意图如图所示。


![](https://files.mdnice.com/user/6935/7842ce3e-a60f-4c37-b520-98851537f16d.png)


图中的蓝色部分，表示需要归一化的部分。其中两维$C$和$N$分别表示$channel$和$batch$  $size$，第三维表示$H$,$W$，可以理解为该维度大小是$H*W$，也就是拉长成一维，这样总体就可以用三维图形来表示。可以看出$BN$的计算和$batch$  $size$相关（蓝色区域为计算均值和方差的单元），而$LN$、$BN$和$GN$的计算和$batch$ $size$无关。同时$LN$和$IN$都可以看作是$GN$的特殊情况（$LN$是$group$=1时候的$GN$，$IN$是$group=C$时候的$GN$）。



### Batch Normalization；

​    $BN$的简单计算步骤为：

- 沿着通道计算每个batch的均值$\mu=\frac{1}{m} \sum_{i=1}^{m} x_{i}$。

- 沿着通道计算每个$batch$的方差$\delta^{2}=\frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2}$。

- 对x做归一化, ![image](https://user-images.githubusercontent.com/47493620/118058914-dd6e8580-b3c1-11eb-8f27-8107dba60cae.png)


- 加入缩放和平移变量$\gamma$和$\beta$ ,归一化后的值，$y_{i} \leftarrow \gamma \widehat{x}_{i}+\beta$


$BN$适用于判别模型中，比如图片分类模型。因为$BN$注重对每个$batch$进行归一化，从而保证数据分布的一致性，而判别模型的结果正是取决于数据整体分布。但是$BN$对$batchsize$的大小比较敏感，由于每次计算均值和方差是在一个$batch$上，所以如果$batchsize$太小，则计算的均值、方差不足以代表整个数据分布。（其代码见下BN的前向与反向的代码详解！）

在训练过程之中，我们主要是通过**滑动平均**这种**Trick**的手段来控制变量更新的速度。

```python
def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Input:
  - x: (N, D)维输入数据
  - gamma: (D,)维尺度变化参数 
  - beta: (D,)维尺度变化参数
  - bn_param: Dictionary with the following keys:
    - mode: 'train' 或者 'test'
    - eps: 一般取1e-8~1e-4
    - momentum: 计算均值、方差的更新参数
    - running_mean: (D,)动态变化array存储训练集的均值
    - running_var：(D,)动态变化array存储训练集的方差

  Returns a tuple of:
  - out: 输出y_i（N，D）维
  - cache: 存储反向传播所需数据
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  # 动态变量，存储训练集的均值方差
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  # TRAIN 对每个batch操作
  if mode == 'train':
    sample_mean = np.mean(x, axis = 0)
    sample_var = np.var(x, axis = 0)
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_hat + beta
    cache = (x, gamma, beta, x_hat, sample_mean, sample_var, eps)
    #滑动平均(影子变量)这种Trick的引入，目的是为了控制变量更新的速度，防止变量的突然变化对变量的整体影响，这能提高模型的鲁棒性。
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
  # TEST：要用整个训练集的均值、方差
  elif mode == 'test':
    x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * x_hat + beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache
```




![image](https://user-images.githubusercontent.com/47493620/118059082-39d1a500-b3c2-11eb-80f8-75f2bf677451.png)
  

  下面来一个背诵版本：

![image](https://user-images.githubusercontent.com/47493620/118059220-9208a700-b3c2-11eb-841f-73781fa93342.png)



因此，$BN$的反向传播代码如下：

```python
def batchnorm_backward(dout, cache):
    """
    Inputs:
    - dout: 上一层的梯度，维度(N, D)，即 dL/dy
    - cache: 所需的中间变量，来自于前向传播

    Returns a tuple of:
    - dx: (N, D)维的 dL/dx
    - dgamma: (D,)维的dL/dgamma
    - dbeta: (D,)维的dL/dbeta
    """
      x, gamma, beta, x_hat, sample_mean, sample_var, eps = cache
      N = x.shape[0]

      dgamma = np.sum(dout * x_hat, axis = 0)
      dbeta = np.sum(dout, axis = 0)

      dx_hat = dout * gamma
      dsigma = -0.5 * np.sum(dx_hat * (x - sample_mean), axis=0) * np.power(sample_var + eps, -1.5)
      dmu = -np.sum(dx_hat / np.sqrt(sample_var + eps), axis=0) - 2 * dsigma*np.sum(x-sample_mean, axis=0)/ N
      dx = dx_hat /np.sqrt(sample_var + eps) + 2.0 * dsigma * (x - sample_mean) / N + dmu / N

      return dx, dgamma, dbeta
```



那么为啥要用$BN$呢？$BN$的作用如下：

- $BN$加快网络的训练与收敛的速度

  在深度神经网络中中，如果每层的数据分布都不一样的话，将会导致网络非常难收敛和训练。如果把每层的数据都在转换在均值为零，方差为1 的状态下，这样每层数据的分布都是一样的训练会比较容易收敛。

- 控制梯度爆炸防止梯度消失

  以$sigmoid$函数为例，$sigmoid$函数使得输出在$[0,1]$之间，实际上当 输入过大或者过小，经过sigmoid函数后输出范围就会变得很小，而且反向传播时的梯度也会非常小，从而导致梯度消失，同时也会导致网络学习速率过慢；同时由于网络的前端比后端求梯度需要进行更多次的求导运算，最终会出现网络后端一直学习，而前端几乎不学习的情况。Batch Normalization (BN) 通常被添加在每一个全连接和激励函数之间，使数据在进入激活函数之前集中分布在0值附近，大部分激活函数输入在0周围时输出会有加大变化。

  同样，使用了$BN$之后，可以使得权值不会很大，不会有梯度爆炸的问题。

- 防止过拟合

  在网络的训练中，BN的使用使得一个$minibatch$中所有样本都被关联在了一起，因此网络不会从某一个训练样本中生成确定的结果，即同样一个样本的输出不再仅仅取决于样本的本身，也取决于跟这个样本同属一个$batch$的其他样本，而每次网络都是随机取$batch$，比较多样，可以在一定程度上避免了过拟合。



### Instance Normalization

IN适用于生成模型中，比如图片风格迁移。因为图片生成的结果主要依赖于某个图像实例，所以对整个$batch$归一化不适合图像风格化中，在风格迁移中使用$Instance $ $Normalization$。不仅可以加速模型收敛，并且可以保持每个图像实例之间的独立。

当然，其前向反向的推导与$BN$相似，无非是维度的问题了～

$Instance$  $ Norm$代码如下所示：

```python
def Instancenorm(x, gamma, beta):
    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5
    x_mean = np.mean(x, axis=(2, 3), keepdims=True)
    x_var = np.var(x, axis=(2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```

### Layer Normalization

$LN$是指对同一张图片的同一层的所有通道进行$Normalization$操作。与上面的计算方式相似，计算均值与方差，在计算缩放和平移变量$\gamma$和$\beta$。$LN$ 主要用在$NLP$任务中，当然，像$Transformer$中存在的就是$LN$。

```python
def Layernorm(x, gamma, beta):
    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5
    x_mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(1, 2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```



**为什么RNN中不使用BN？**

$RNN$可以展开成一个隐藏层共享参数的$MLP$，随着时间片的增多，展开后的$MLP$的层数也在增多，最终层数由输入数据的时间片的数量决定，所以$RNN$是一个动态的网络。在$RNN$网络中，一个$batch$的数据中，通常各个样本的长度都是不同的。往往在最后时刻，只有少量样本有数据，基于这个样本的统计信息不能反映全局分布，所以这时$BN$的效果并不好。

而当将$LN$添加到$CNN$之后，实验结果发现$LN$破坏了卷积学习到的特征，模型无法收敛，所以在$CNN$之后使用$BN$是一个更好的选择。

对于$LN$与$BN$ 而言，$BN$取的是不同样本的同一个特征，而$LN$取的是同一个样本的不同特征。在$BN$和$LN$都能使用的场景中，$BN$的效果一般优于$LN$，原因是基于不同数据，同一特征得到的归一化特征更不容易损失信息。但是有些场景是不能使用$BN$的，例如$batchsize$较小或者在$RNN$中，这时候可以选择使用$LN$，$LN$得到的模型更稳定且起到正则化的作用。$RNN$能应用到小批量和$RNN$中是因为$LN$的归一化统计量的计算是和$batchsize$没有关系的。

### Group Normalization

$Group$  $Normalization$（$GN$）是针对$Batch$ $ Normalization$（BN）在$batch$ $ size$较小时错误率较高而提出的改进算法，因为$BN$层的计算结果依赖当前$batch$的数据，当$batch$  $ size$较小时（比如2、4这样），该$batch$数据的均值和方差的代表性较差，因此对最后的结果影响也较大。

其中，$GN$是将通道数$C$分成$G$份，每份$C//G$，当$G=1$时，每份$G$个，所以为一整块的$C$,即为$LN$ 。当$G=C$时，每份只有$1$个，所以为$IN$。

$GN$是指对同一张图片的同一层的某几个（不是全部）通道一起进行$Normalization$操作。这几个通道称为一个$Group$。计算相应的均值以及方差，计算缩放和平移变量$\gamma$和$\beta$。
其代码如下所示：

```python
def GroupNorm(x, gamma, beta, G=16):

    # x_shape:[B, C, H, W]
    # gamma， beta, scale, offset : [1, c, 1, 1]
    # G: num of groups for GN
    results = 0.
    eps = 1e-5
    x = np.reshape(x, (x.shape[0], G, x.shape[1]/16, x.shape[2], x.shape[3]))

    x_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(x, axis=(2, 3, 4), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```



### Switchable Normalization

$SN$中表明了一个观点：$BN$层的成功和协方差什么的没有关联！证明这种层输入分布稳定性与 $BN$ 的成功几乎没有关系。相反，实验发现 $BN$ 会对训练过程产生更重要的影响：它使优化解空间更加平滑了。这种平滑使梯度更具可预测性和稳定性，从而使训练过程更快。

$SN$的具体做法可以从图中看出来：

![](https://files.mdnice.com/user/6935/e53e9385-0507-4d4b-a84a-86cbd6490cee.png)


论文中认为：

- 第一，归一化虽然提高模型泛化能力，然而归一化层的操作是人工设计的。在实际应用中，解决不同的问题原则上需要设计不同的归一化操作，并没有一个通用的归一化方法能够解决所有应用问题；

- 第二，一个深度神经网络往往包含几十个归一化层，通常这些归一化层都使用同样的归一化操作，因为手工为每一个归一化层设计操作需要进行大量的实验。

而与强化学习不同，$SN$使用可微分学习，为一个深度网络中的每一个归一化层确定合适的归一化操作。



$SN$算法是为三组不同的 $\mu_{k}$ 以及$\sigma_{k}$ 分别学习三个总共6个标量值（ $w_{k}$ 和$w_{k}^{\prime}$ )，$h_{n c i j}$表示一个输入维度为$n,c,i,j$的特征图,$\hat{h}_{n c i j}$为归一化之后的特征图。

![image](https://user-images.githubusercontent.com/47493620/118059370-f9265b80-b3c2-11eb-956a-d5ebe72b0e0c.png)

其中 $\Omega=\{i n, l n, b n\}$ 。在计算 $(\mu_{\ln }, \sigma_{\ln })$ 和 $(\mu_{b n}, \sigma_{b n})$ 时，我们可以使用 $(\mu_{i n}, \sigma_{i n})$ 作为中间变量以减少计算量。

$\mu_{\mathrm{in}}=\frac{1}{H W} \sum_{i, j}^{H, W} h_{n c i j}, \quad \sigma_{\mathrm{in}}^{2}=\frac{1}{H W} \sum_{i, j}^{H, W}\left(h_{n c i j}-\mu_{\mathrm{in}}\right)^{2}$
$\mu_{\mathrm{ln}}=\frac{1}{C} \sum_{c=1}^{C} \mu_{\mathrm{in}}, \quad \sigma_{\mathrm{ln}}^{2}=\frac{1}{C} \sum_{c=1}^{C}\left(\sigma_{\mathrm{in}}^{2}+\mu_{\mathrm{in}}^{2}\right)-\mu_{\mathrm{ln}}^{2}$
$\mu_{\mathrm{bn}}=\frac{1}{N} \sum_{n=1}^{N} \mu_{\mathrm{in}}, \quad \sigma_{\mathrm{bn}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(\sigma_{\mathrm{in}}^{2}+\mu_{\mathrm{in}}^{2}\right)-\mu_{\mathrm{bn}}^{2}$



$w_{k}$ 是通过softmax计算得到的激活函数：

$w_{k}=\frac{e^{\lambda_{k}}}{\sum_{z \in \mathrm{in}, \ln , \mathrm{bn}} e^{\lambda_{z}}} \quad$ and $\quad k \in\{\mathrm{in}, \ln , \mathrm{bn}\}$

![image](https://user-images.githubusercontent.com/47493620/118059543-5c17f280-b3c3-11eb-96a6-d85bf66c2026.png)


代码如下：



```python
def SwitchableNorm(x, gamma, beta, w_mean, w_var):
    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    mean_in = np.mean(x, axis=(2, 3), keepdims=True)
    var_in = np.var(x, axis=(2, 3), keepdims=True)

    mean_ln = np.mean(x, axis=(1, 2, 3), keepdims=True)
    var_ln = np.var(x, axis=(1, 2, 3), keepdims=True)

    mean_bn = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var_bn = np.var(x, axis=(0, 2, 3), keepdims=True)

    mean = w_mean[0] * mean_in + w_mean[1] * mean_ln + w_mean[2] * mean_bn
    var = w_var[0] * var_in + w_var[1] * var_ln + w_var[2] * var_bn

    x_normalized = (x - mean) / np.sqrt(var + eps)
    results = gamma * x_normalized + beta
    return results
```

值得一提的是在测试的时候，在$SN$的$BN$部分，它使用的是一种叫做**批平均（**batch average)的方法，它分成两步：1.固定网络中的$SN$层，从训练集中随机抽取若干个批量的样本，将输入输入到网络中；2.计算这些批量在特定SN层的 $\mu$ 和 $\sigma$ 的平均值，它们将会作为测试阶段的均值和方差。实验结果表明，在$SN$中批平均的效果略微优于滑动平均。



## 2. 对模型权重做归一化

### Weight Normalization

$Weight$    $ Normalization（WN)$是在权值的维度上做的归一化。$WN$做法是将权值向量 $w$在其欧氏范数和其方向上解耦成了参数向量 $v$和参数标量$g$ 后使用$SGD$分别优化这两个参数。

$WN$也是和样本量无关的，所以可以应用在$batchsize$较小以及$RNN$等动态网络中；另外$BN$使用的基于$mini-batch$的归一化统计量代替全局统计量，相当于在梯度计算中引入了噪声。而$WN$则没有这个问题，所以在生成模型与强化学习等噪声敏感的环境中$WN$的效果也要优于$BN$。



$WN$的计算过程：

对于神经网络而言，一个节点的计算过程可以表达为：

$$y=\phi(\mathbf{w} \cdot \mathbf{x}+b)$$

其中$w$是与该神经元连接的权重，通过损失函数与梯度下降对网络进行优化的过程就是求解最优$w$的过程。将$w$的**长度**与**方向**解耦，可以将$w$表示为:
$$
w=\frac{g}{\|v\|} v
$$
$y=\phi(w * x+b)$
$w$ 为与该神经元连接的权重，通过损失函数与梯度下降对网络进行优化的过程就是求解最优 $w$ 的
过程。将 $w$ 的长度与方向解塊，可以将 $w$ 表示为
$w=\frac{g}{\|v\|} v$

其中 $g$ 为标量，其大小等于 $w$ 的模长, $\frac{v}{\|v\|}$ 为与 $w$ 同方向的单位向量，此时，原先训练过程 中 $w$ 的学习转化为 $g$ 和 $v$ 的学习。假设损失函数以 $L$ 表示，则 $L$ 对 $g$ 和 $v$ 的梯度可以分别
表示为,
$\nabla_{g} L=\nabla_{g} w *\left(\nabla_{w} L\right)^{T}=\frac{\nabla_{w} L * v^{T}}{\|v\|}$
$\nabla_{v} L=\nabla_{v} w * \nabla_{w} L=\frac{\partial \frac{g * v}{\|v\|}}{\partial v} * \nabla_{w} L=\frac{g *\|v\|}{\|v\|^{2}} * \nabla_{w} L-\frac{g * v * \frac{\partial\|v\|}{\partial v}}{\|v\|^{2}} * \nabla_{w} L$
因为
$\frac{\partial\|v\|}{v}=\frac{\partial\left(v^{T} * v\right)^{0.5}}{\partial v}=0.5 *\left(v^{T} * v\right)^{-0.5} * \frac{\partial\left(v^{T} * v\right)}{\partial v}=\frac{v}{\|v\|}$
所以
$\nabla_{v} L=\frac{g}{\|v\|} * \nabla_{w} L-\frac{g * \nabla_{g} L}{\|v\|^{2}} * v=\frac{g}{\|v\|} * M_{w} * \nabla_{w} L$

其中 $M_{w}=I-\frac{w * w^{T}}{\|w\|^{2}}$，与向量点乘可以投影任意向量至 $w$ 的补空间。相对于原先的$\nabla_{w} L, \quad \nabla_{v} L$进行了$\frac{g}{\|v\|}$的缩放与$M_{w}$的投影。这两者都对优化过程起到作用。

对于$v$而言，$v^{\text {new }}=v+\Delta v$, 因为 $\Delta v \propto \nabla_{v} L$ ，所以 $\Delta v$ 与 $v$ 正交。

假设，$\frac{\|\Delta v\|}{\|v\|}=c$, 则 $\left\|v^{n e w}\right\|=\sqrt{\|v\|^{2}+c^{2}\|v\|^{2}}=\sqrt{1+c^{2}}\|v\| \geq\|v\|, \quad \nabla_{v} L$ 可以影响$v$ 模长的增长，同时 $v$ 的模长也影响 $\nabla_{v} L$ 的大小。

因此，我们可以得到：

- $\frac{g}{\|\mathbf{v}\|}$表明$WN$会对权重梯度进行$\frac{g}{\|\mathbf{v}\|}$的缩放。
- $M_{\mathbf{w}} \nabla_{\mathbf{w}} L$表明WN会将梯度投影到一个远离于$\nabla_{\mathbf{w}} L$的方向。

代码可以参考：

```python
import torch.nn as nn
import torch.nn.functional as F

# 以一个简单的单隐层的网络为例
class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(Model, self).__init__()
        # weight_norm
        self.dense1 = nn.utils.weight_norm(nn.Linear(input_dim, hidden_size))
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, output_dim))
    
    def forward(self, x):
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        return x
```



### Spectral Normalization

首先看下这个图，了解下一个数学概念叫做**Lipschitz 连续性**：


![](https://files.mdnice.com/user/6935/6db9e006-a0e4-4f41-9675-2f8796565c1d.png)


$Lipschitz$ 条件限制了函数变化的剧烈程度，即函数的梯度。在一维空间中，很容易看出$ y=sin(x)$ 是$ 1-Lipschitz$的，它的最大斜率是 1。

如$y=x$与$y=-x$的斜率是$1$与$-1$,$sin(x)$求导是$cosx(x)$，值域为$[0,1]$.



在$GAN$中，假设我们有一个判别器 $D: I \rightarrow \mathbb{R}$ ，其中 $I$是图像空间。如果判别器是 $K-Lipschitz continuous$ 的，那么对图像空间中的任意$ x$ 和$ y$，有：
$$
\|D(x)-D(y)\| \leq K\|x-y\|
$$
其中$\|\cdot\|$为$L_{2}$ norm,如果$ K$ 取到最小值，那么$ K$ 被称为$ Lipschitz$   $ constant$。

首先抛出结论：

**矩阵$A$除以它的 spectral norm ($A^{T} A$最大特征值的开根号$\sqrt{\lambda_{1}}$）可以使其具有$ 1-Lipschitz$ $ continuity$**。

那么$Spectral$  $Normalization$的具体做法如下：

- 将神经网络的每一层的参数 $W$ 作$ SVD$ 分解，然后将其最大的奇异值限定为$1$，满足$1-Lipschitz$条件。

  在每一次更新$W$之后都除以$W$最大的奇异值。 这样，每一层对输入 $x$ 最大的拉伸系数不会超过 $1$。

  经过$Spectral$ $ Norm$ 之后，神经网络的每一层$g_{l}(x)$ 权重，都满足
  $$
  \frac{g_{l}(x)-g_{l}(y)}{x-y} \leq 1
  $$
  对于整个神经网络 $f(x)=g_{N}\left(g_{N-1}\left(\ldots g_{1}(x) \ldots\right)\right)$自然也就满足利普希茨连续性了

- 在每一次训练迭代中，都对网络中的每一层都进行$SVD$分解，是不现实的，尤其是当网络权重维度很大的时候。我们现在可以使用一种叫做$power$ $iteration$的算法。

  $Power$ $ iteration$ 是用来近似计算矩阵最大的特征值（$dominant$ $ eigenvalue$ 主特征值）和其对应的特征向量（主特征向量）的。

  假设矩阵$A$是一个$n$ x $n$的满秩的方阵，它的单位特征向量为$v_{1}, v_{2}, … v_{n}$，对于的特征值为$\lambda_{1}, \lambda_{2}, …\lambda_{n}$。那么任意向量$x=x_{1} * v_{1} + x_{2} * v_{2} + … x_{n} * v_{n}$。则有：
  $$
  \begin{aligned}
  A x &=A\left(x_{1} \cdot \nu_{1}+x_{2} \cdot \nu_{2}+\ldots+x_{n} \cdot \nu_{n}\right) \\
  &=x_{1}\left(A \nu_{1}\right)+x_{2}\left(A \nu_{2}\right)+\ldots+x_{n}\left(A \nu_{n}\right) \\
  &=x_{1}\left(\lambda_{1} \nu_{1}\right)+x_{2}\left(\lambda_{2} \nu_{2}\right)+\ldots+x_{n}\left(\lambda_{n} \nu_{n}\right)
  \end{aligned}
  $$
  我们通过$k$次迭代：
  $$
  \begin{aligned}
  A^{k} x &=x_{1}\left(\lambda_{1}^{k} \nu_{1}\right)+x_{2}\left(\lambda_{2}^{k} \nu_{2}\right)+\ldots+x_{n}\left(\lambda_{n}^{k} \nu_{n}\right) \\
  &=\lambda_{1}^{k}\left[x_{1} \nu_{1}+x_{2}\left(\frac{\lambda_{2}}{\lambda_{1}}\right)^{k} \nu_{2}+\ldots+x_{n}\left(\frac{\lambda_{n}}{\lambda_{1}}\right)^{k} \nu_{n}\right]
  \end{aligned}
  $$
  ![image](https://user-images.githubusercontent.com/47493620/118059645-971a2600-b3c3-11eb-9a8a-81f4249082aa.png)


同样，我们可以得到：

![](https://files.mdnice.com/user/6935/af0fae69-aa01-42c1-bc7d-6f9243a528cf.png)


具体的代码实现过程中，可以随机初始化一个噪声向量代入公式 (13) 。由于每次更新参数的$ step$ $ size$ 很小，矩阵 W 的参数变化都很小，矩阵可以长时间维持不变。

因此，可以把参数更新的 $step$ 和求矩阵最大奇异值的$ step$ 融合在一起，即每更新一次权重$ W$ ，更新一次和，并将矩阵归一化一次。

代码如下：

```python
import torch
from torch.optim.optimizer import Optimizer, required

import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
        def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

        def forward(self, *args):
          self._update_u_v()
          return self.module.forward(*args)

```







​            
