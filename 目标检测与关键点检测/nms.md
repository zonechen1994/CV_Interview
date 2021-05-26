### 1. $NMS$代码与实现

**$Non$-$Maximum$-$Suppression$(非极大值抑制**): 当两个$box$空间位置非常接近，就以$score$更高的那个作为基准，看$IOU$即重合度如何，如果与其重合度超过阈值，就抑制$score$更小的$box$，只保留$score$大的就$Box$，其它的$Box$就都应该过滤掉。对于$NMS$而言，适合于水平框，针对各种不同形状的框，会有不同的$NMS$来进行处理。

**具体的步骤如下**：

![image](https://user-images.githubusercontent.com/47493620/119690666-78646680-be7c-11eb-8499-d757d6b7fa66.png)


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
![image](https://user-images.githubusercontent.com/47493620/119690751-8b773680-be7c-11eb-88d3-4757632e1904.png)
