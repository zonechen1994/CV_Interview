## 面试必考 $NMS$汇总
### 1. $NMS$代码与实现

**$Non$-$Maximum$-$Suppression$(非极大值抑制**): 当两个$box$空间位置非常接近，就以$score$更高的那个作为基准，看$IOU$即重合度如何，如果与其重合度超过阈值，就抑制$score$更小的$box$，只保留$score$大的就$Box$，其它的$Box$就都应该过滤掉。对于$NMS$而言，适合于水平框，针对各种不同形状的框，会有不同的$NMS$来进行处理。

**具体的步骤如下**：


![](https://files.mdnice.com/user/6935/d16d78d9-7595-44bf-895c-4d88902c7c2c.jpg)


1. 如图所示，我们有$6$个带置信率的$region$ $proposals$，我们先预设一个$IOU$的阈值如$0.7$。
2. 按置信率大小对$6$个框排序，举例为 $0.94, 0.91, 0.90, 0.83, 0.79, 0.77$。
3. 设定置信率为$0.94$的$region$ $proposals$为一个物体框；
4. 在剩下$5$个$region$ $proposals$中进行循环遍历，去掉与$0.94$物体框$IOU$大于$0.7$的。
5. 重复$2$～$4$的步骤，直到没有$egion$ $proposals$为止。
6. 每次获取到的最大置信率的$region$ $proposals$就是我们筛选出来的目标。

参考代码如下：

```python
import numpy as np

def NMS(dets, thresh):
    """Pure Python NMS baseline."""
    # tl_x,tl_y,br_x,br_y及score
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #计算每个检测框的面积，并对目标检测得分进行降序排序
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []   #保留框的结果集合
    while order.size > 0:
        i = order[0]
        keep.append(i)　　#保留该类剩余box中得分最高的一个
        # 计算最高得分矩形框与剩余矩形框的相交区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

       #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]   #注意这里索引加了1,因为ovr数组的长度比order数组的长度少一个

    return keep
```

运行后，则删除了多余的框，结果如图所示：


![](https://files.mdnice.com/user/6935/701bb3e8-0852-48fa-ad41-0ad7405d1c02.jpg)


### 2. $Soft$ $NMS$的代码与实现

说到$Soft$ $NMS$，首先需要了解传统$NMS$有哪些缺点。其主要缺点包括如下：

- **物体重叠**：如下面第一张图，会有一个最高分数的框，如果使用$NMS$的话就会把其他置信度稍低，但是表示另一个物体的预测框删掉（由于和最高置信度的框$overlap$过大）


![](https://files.mdnice.com/user/6935/3c3b7605-fb13-4fae-a4f1-890f58e3f52d.png)


- **所有的$bbox$都预测不准**：不是所有的框都那么精准，有时甚至会出现某个物体周围的所有框都标出来了，但是都不准的情况，如下图所示。


![](https://files.mdnice.com/user/6935/a85cd387-07aa-48aa-9bc2-1d08091d8bfe.jpg)


- 传统的$NMS$方法是基于分类分数的，**只有最高分数的预测框能留下来，但是大多数情况下$IoU$和分类分数不是强相关**，很多分类标签置信度高的框都位置都不是很准。


![](https://files.mdnice.com/user/6935/53b31a1c-b3ee-4cf3-a89b-4387c35ef46a.png)


$Soft$ $NMS$主要是针对$NMS$过度删除框的问题。$Soft-NMS$吸取了$NMS$的教训，在算法执行过程中不是简单的对$IoU$大于阈值的检测框删除，而是降低得分。算法流程同$NMS$相同，但是对原置信度得分使用函数运算，目标是降低置信度得分。其算法步骤如下：


![](https://files.mdnice.com/user/6935/5b7c1277-7f4a-425c-9282-028af50cc829.png)


红色的部分表示原始$NMS$算法，绿色部分表示$Soft$-$NMS$算法，区别在于，绿色的框只是把$s_{i}$降低了，而不是把$b_{i}$直接去掉，极端情况下，如果$f$只返回$0$，那么等同于普通的$NMS$。

$b_{i}$为待处理$BBox$框，$B$为待处理$BBox$框集合，$s_{i}$是$b_{i}$框更新得分，$N_{t}$是$NMS$的阈值，$D$集合用来放最终的$BBox$，$f$是置信度得分的重置函数。$b_{i}$和$M$的$IOU$越大，$b_{i}$的得分$s_{i}$就下降的越厉害。

$f$函数是为了降低目标框的置信度，满足条件，如果$b_{i}$和$M$的$IoU$越大，$f(iou(M, bi))$就应该越小，$Soft$-$NMS$提出了两种$f$函数：

经典的$NMS$算法将$IOU$大于阈值的窗口的得分全部置为$0$，可表述如下：


![](https://files.mdnice.com/user/6935/77614fcf-2e82-4ea8-b56d-a5adc4477a6d.png)




论文置信度重置函数有两种形式改进，**一种是线性加权的**：


![](https://files.mdnice.com/user/6935/3ba5aa4e-0473-41b1-8e06-0c5ccde66722.png)




**一种是高斯加权形式**：

$s_{i}=s_{i} e^{-\frac{\mathrm{iou}\left(\mathcal{M}, b_{i}\right)^{2}}{\sigma}}, \forall b_{i} \notin \mathcal{D}$



$Soft $ $NMS$算法的优点如下：

- 该方案可以很方便地引入到object detection算法中，不需要重新训练原有的模型;
- **soft-NMS在训练中采用传统的NMS方法，可以仅在推断代码中实现soft-NMS**。

- $NMS$是$Soft$-$NMS$特殊形式，当得分重置函数采用二值化函数时，$Soft$-$NMS$和$NMS$是相同的。$soft$-$NMS$算法是一种更加通用的非最大抑制算法。

而，在一些场景的实验中，可以看到$Soft$ $NMS$的效果也是优于$NMS$的。

![img](http://princepicbed.oss-cn-beijing.aliyuncs.com/blog_20181201144831.png)

这里提供一个$github$ 中的$Cython$代码展示:

```python
def cpu_soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):
    cdef unsigned int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
 
    for i in range(N):
        
        # 在i之后找到confidence最高的框，标记为max_pos
        maxscore = boxes[i, 4]
        maxpos = i
 
        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]
 
        pos = i + 1
	    # 找到max的框
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1
        
        # 交换max_pos位置和i位置的数据
	    # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]
 
	    # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts
 
        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]
        # 交换完毕
        
        # 开始循环
        pos = i + 1
        
        while pos < N:
            # 先记录内层循环的数据bi
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]
            
            # 计算iou
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1) # 计算两个框交叉矩形的宽度，如果宽度小于等于0，即没有相交，因此不需要判断
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1) # 同理
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih) #计算union面积
                    ov = iw * ih / ua #iou between max box and detection box
 
                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1
 
                    boxes[pos, 4] = weight*boxes[pos, 4]
		    
		            # if box score falls below threshold, discard the box by swapping with last box
		            # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1
 
            pos = pos + 1
 
    keep = [i for i in range(N)]
    return keep
```



### $Softer$ $NMS$的代码与实现

针对剩余的两个问题，$Softer$ $NMS$做出了自己的努力。

- 针对分类置信度和框的$IoU$不是强相关的问题，构建一种$IoU$的置信度，来建模有多大把握认为当前框和$GT$是重合的。

- 针对所有的框单独拿出来都不准的问题，文章中提出一种方法，根据$IoU$置信度加权合并多个框优化最终生成框。

$Softer$-$NMS$文章对预测框建模，以下公式中$x$表示偏移前的预测框，$x_{e}$表示偏移后的预测框，输出的$x_{g}$表示$GT$框，使用高斯函数对预测框建模:
$$
P_{\Theta}(x)=\frac{1}{2 \pi \sigma^{2}}e^{-\frac{(x-x_{e})^2}{2 \sigma^{2}}}
$$
对于$GT$框建模：使用$delta$分布(**即标准方差为$0$的高斯分布极限**)。
$$
P_{D}(x)=\delta\left(x-x_{g}\right)
$$
对于$delta$分布，当$\sigma$越小，其函数图像就会越瘦高，同时，当$\sigma$越小，表示网络越确定，可以使用$1-\sigma$就可以作为网络的置信度。

同时，论文使用$KL$散度来最小化$Bounding$ $box$ $regression$ $loss$。既$Bounding$ $box$的高斯分布和$ground$ $truth$的狄拉克$delta$分布的$KL$散度。直观上解释，$KL$ $Loss$使得$Bounding$ $box$预测呈高斯分布，且与$ground$ $truth$相近。而将包围框预测的标准差看作置信度。



如$faster$ $rcnn$中添加了$softer$ $nms$之后的示意图如图所示：


![](https://files.mdnice.com/user/6935/7f695b9d-bb97-4327-b9b4-00cd165d21a5.png)


多加了一个$\sigma$预测，也就是$box$ $std$，而$Box$的预测其实就是上面公式中的$x_{e}$。

因此，整个计算过程如下：

1. 计算$x_{e}$与$x$的2范数距离和$\sigma$计算出$P_{\theta}(x)$.
2. 通过$x_{g}$与$x$的2范数距离算出$P_{D}$.
3. 使用$P_{D}$与$P_{\theta}$计算$KLs$散度作为$loss$，最小化$KLLoss$。

关于坐标回归的损失函数：
$$
\begin{array}{l}
L_{r e g}=D_{K L}\left(P_{D}(x) \| P_{\Theta}(x)\right) \\
=\int P_{D}(x) \log \frac{P_{D}(x)}{P_{\Theta}(x)} d x \\
=-\int P_{D}(x) \log P_{\Theta}(x) d x+\int P_{D}(x) \log P_{D}(x) d x \\
=-\int P_{D}(x) \log P_{\Theta}(x) d x+H\left(P_{D}(x)\right) \\
=-\log P_{\Theta}\left(x_{g}\right)+H\left(P_{D}(x)\right) \\
=\frac{\left(x_{g}-x_{e}\right)^{2}}{2 \sigma^{2}}+\frac{1}{2} \log \left(\sigma^{2}\right)+\frac{1}{2} \log (2 \pi)+H\left(P_{D}(x)\right)
\end{array}
$$
而后面两项是与$x_{e}$无关，可以去掉～
$$
L_{\text {reg }}=\alpha\left(\left|x_{g}-x_{e}\right|-\frac{1}{2}\right)-\frac{1}{2} \log (\alpha+\epsilon)
$$
因此，计算过程如下图所示：

![](https://files.mdnice.com/user/6935/7ac68b67-df56-425d-a7b3-6c4d35e90d5c.png)


网络预测出来的结果是$x1_{i}, y1_{i}, x2_{i}, y2_{i}, \sigma{x1_{i}}, \sigma{x2_{i}}, \sigma{x3_{i}}, \sigma{x4_{i}}$。前面四个为坐标，而后面四个是坐标的$\sigma$。

上表中的蓝色的是$soft$-$nms$，只是降低了$S$的权值。重点看绿色的，绿字第一行表示拿出所有与$B$的$IoU$大于$N_{t}$的框（用$idx$表示），然后将所有这些框做一个加权，$B[idx]/C[idx]$其实是$B[idx] * 1/C[idx]$，后者是置信度$\frac{1}{\sigma^{2}}$，并使用$sum$做了归一化。需要注意的是，$Softer$-$NMS$算法中，$B$是不变的，$softer$-$nms$只调整每个框的位置，而不筛选框。

贴一张效果图：


![](https://files.mdnice.com/user/6935/a4b87cf8-1881-4942-bf40-4b76b2803fb9.png)

