## CNN

- 局部连接：不是全连接，而是使用size相对input小的kernel在局部感受视野内进行连接（点积运算）
- 权值共享：在一个卷积核运算中，每次都运算一个感受视野，通过滑动遍历的把整个输入都卷积完成，而不是每移动一次就更换卷积核参数



两者目的都是减少参数。通过局部感受视野，通过卷积操作获取高阶特征，能达到比较好的效果。

### 内部计算
先看卷积（卷积核与感受视野的点积）与池化示意图：

![](https://files.mdnice.com/user/6935/022ed39e-a493-4f7b-8a3a-9476dcfc3c3f.gif)

最大池化层:

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWcyMDE4LmNuYmxvZ3MuY29tL2Jsb2cvOTkxNDcwLzIwMTkwMi85OTE0NzAtMjAxOTAyMDgyMDE1MDg3MDQtMzY4NjQ0NzkyLnBuZw?x-oss-process=image/format,png)

- 输入、卷积、池化形状定义:
   输入为（长，宽，RGB通道数）=（7*7*3）的图片，即inputs = (batch_size, 7, 7, 3)
  卷积核为（长，宽，卷积通道数，卷积核个数），即filter = (3, 3, 3, 32)。
  池化为（长，宽，步长）
  - 标准卷积运算过程：
    1. 对于输入没有疑问，对于卷积操作，定义的32个卷积核，每一个卷积核都有3个通道，相同的卷积核在三个通道的卷积核参数不相同，只有尺寸相同，每个卷积核都有3个尺寸但参数不一样的变量，最后该卷积核的结果为每个通道的卷积结果相加，得到卷积后的新通道。卷积核的个数为卷积完成后新的通道数。
    2. 参数个数为 32*3， 参数最后的size为3x3x3x32。

- 池化的意义

  1.特征不变形：池化操作是模型更加关注是否存在某些特征而不是特征具体的位置。

　　2.特征降维：池化相当于在空间范围内做了维度约减，从而使模型可以抽取更加广范围的特征。同时减小了下一层的输入大小，进而减少计算量和参数个数。

　　3.在一定程度上防止过拟合，更方便优化。



## 卷积和池化后的大小计算 

给定的值：原大小 $(H, W)$, 卷积核大小 $(h, w)$, 池化大小 $(h, w)$, padding(填充)=a, strids (滑动步长） $=\mathrm{b}$, 公式通用 计算公式: $\left\{\begin{array}{c}H^{\prime}=\frac{H+2 * a-h}{b}+1 \\ W^{\prime}=\frac{W+2 * a-w}{b}+1\end{array}\right.$





假设：输入图片大小为200×200，依次经过一层卷积（kernel size 5×5，padding 1，stride 2），pooling（kernel size 3×3，padding 0，stride 1），又一层卷积（kernel size 3×3，padding 1，stride 1）之后，输出特征图大小为：

1. 卷积后: $H^{\prime}=\frac{200+2 * 1-5}{2}+1=\frac{197}{2}+1 \approx 99$
2. 池化后: $H^{\prime}=\frac{99+2 * 0-3}{1}+1=\frac{96}{1}+1 \approx 97$
最后大小为 $(97,97)$