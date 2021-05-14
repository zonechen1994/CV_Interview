## 手写目标检测与语义分割中的IOU




[阅读原文](https://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247486222&idx=1&sn=ec8991c9bf1fa2646c6b5f192c4b92f5&chksm=c241e842f53661541d7cb53c2ca828979e5434b4db207c8469d2ee156b4416a667b90841a1b9&scene=178&cur_album_id=1860258784426672132#rd)


大家好，我是灿视。

今天给大家带来两道纯工程的题，是一位博士在面试**face**++时，被问到的。

**看文章之前，别忘了关注我们，在我们这里，有你所需要的干货哦！**



### 1. 目标检测中的IOU



假设，我们有两个框，$rec1$与$rec2$，我们要计算其$IOU$。其中$IOU$的计算公式为，其交叉面积$Intersection$除以其并集$Union$。


![](https://files.mdnice.com/user/6935/9ee39aa5-ba2b-4ab8-821b-337517e00c79.png)


$IOU$的数学公式为：
$$
I o U=\frac{S_{rec1} \cap S_{rec2}}{S_{rec1} + S_{rec2} - S_{rec1} \bigcap S_{rec2}}
$$





上代码：

```python
def compute_iou(rec1, rec2):
    """
    computing IoU
    param rec1: (y0, x0, y1, x1) , which reflects (top, left, bottom, right)
    param rec2: (y0, x0, y1, x1) , which reflects (top, left, bottom, right)
    return : scale value of IoU
    """
    S_rec1 =(rec1[2] -rec1[0]) *(rec1[3] -rec1[1])
    S_rec2 =(rec2[2] -rec2[0]) *(rec2[3] -rec2[1])
    #computing the sum area
    sum_area =S_rec1 +S_rec2
    #find the each edge of interest rectangle
    left_line =max(rec1[1], rec2[1])
    right_line =min(rec1[3], rec2[3])
    top_line =max(rec1[0], rec2[0])
    bottom_line =min(rec1[2], rec2[2])
    #judge if there is an intersect
    if left_line >=right_line or top_line >=bottom_line:
            return 0
    else:
            intersect =(right_line -left_line) +(bottom_line -top_line)
            return intersect /(sum_area -intersect)
```

这里我们主要讨论下这个$if$判断，我们以横轴$x$方向为例，其中对$y$纵轴方向是一样的，我们来判断两个框重合与否。其中$x_{0}$为$rec1$左上角的$x$坐标，$x_{1}$是$rec1$右下角的$x$坐标。$A_{0}$为$rec2$的左上角$x$坐标，$A_{1}$是$rec2$的右下角$x$坐标。

![](https://files.mdnice.com/user/6935/6673500b-e928-4e08-9db8-0f575c98ef58.png)

### 2. 语义分割中的IOU

先回顾下一些基础知识：

常常将预测出来的结果分为四个部分：$true$ $ positive$,$false$ $positive$,$true$ $negative$,$false$ $negative$,其中$negative$就是指非物体标签的部分(可以直接理解为背景)，positive$就是指有标签的部分。下图显示了四个部分的区别：


![](https://files.mdnice.com/user/6935/d4d901fb-7394-4311-8fb6-e59d86817425.png)


$prediction$图被分成四个部分，其中大块的白色斜线标记的是$true$ $negative$（TN，预测中真实的背景部分），红色线部分标记是$false$ $negative$（$FN$，预测中被预测为背景，但实际上并不是背景的部分），蓝色的斜线是$false$ $positive$（$FP$，预测中分割为某标签的部分，但是实际上并不是该标签所属的部分），中间荧光黄色块就是$true$  $positive$（$TP$，预测的某标签部分，符合真值）。

同样的，$IOU$计算公式：
$$
IOU = \frac{\text { target } \bigwedge \text { prediction }}{target \bigcup prediction}
$$

![](https://files.mdnice.com/user/6935/bba60b3c-e8bc-4764-acb1-75d6f5c012de.png)


```py
def compute_ious(pred, label, classes):
    '''computes iou for one ground truth mask and predicted mask'''
    ious = [] # 记录每一类的iou
    for c in classes:
        label_c = (label == c) # label_c为true/false矩阵
        pred_c = (pred == c)
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union == 0:
            ious.append(float('nan'))  
        else
            ious.append(intersection / union)
    return np.nanmean(ious) #返回当前图片里所有类的mean iou
```

其中，对于$label$与$pred$有多种形式。



如识别目标为4类，那么$label$的形式可以是一张图片对应一份$mask[0，1，2，3，4]$，其中$0$ 为背景，我们省略，则$class$可以为$[1,2,3,4]$。也可以是对应四份二进制$mask$ $[0，1]$, 这四层$mask$的取值为$0/1$。$class$为$[1]$了。





### 总结

对于目标检测，写$IOU$那就是必考题，但是我们也要回顾下图像分割的$IOU$怎么计算的。





### 其它干货

- [算法岗，不会写简历？我把它拆开，手把手教你写！](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485095&idx=1&sn=b3fa4c5e87d2c883e4234a512b03f925&chksm=c241e5ebf5366cfd0e1e878d6f81cc441c39da645f53f470547a6e1ca8fad20d3de16f3055bb&scene=21#wechat_redirect)
- [(算法从业人员必备！)Ubuntu办公环境搭建！](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485184&idx=1&sn=cc9ac830e1fccceac03b1ec18c4cdc84&chksm=c241e44cf5366d5ac977c3f78b2b83148a6dba80ab8213c31ecc77582fe2eb2d2991bb76ecfc&scene=21#wechat_redirect)
- [“我能分清奥特曼们了，你能分清我的口红吗？”](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485606&idx=1&sn=a54673568dda61af44ff3a707dd52927&chksm=c241ebeaf53662fc27913f4ce84252efd7d996e16a30828d52dcd840de0868f2ae8f911dda09&scene=21#wechat_redirect)
- [入门算法，看这个呀！(资料可下载)](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485678&idx=1&sn=1f4c265a29bc78f3c3470cdf328a2d7b&chksm=c241eba2f53662b487a3a0a629d97b1e811552153728031c2b30614aeadd722cc83bf1d3d866&scene=21#wechat_redirect)
- [放弃大厂算法Offer，去银行做开发，现在...](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485716&idx=1&sn=ca48d6fd590c9a76749c41c47e5f2da3&chksm=c241ea58f536634e7b19eab8b6f14953e068b8701623fd8c1f3deb6e1abd26503e7062bddcfd&scene=21#wechat_redirect)
- [超6k字长文，带你纵横谈薪市场（建议工程师收藏！)](http://mp.weixin.qq.com/s?__biz=MzkzNDIxMzE1NQ==&mid=2247485766&idx=1&sn=e8c91387c1f8cb5902b695e73018a609&chksm=c241ea0af536631c7c9f01eac9e596536f1c666a824b6ea80915189b773473dd9e54ef26d751&scene=21#wechat_redirect)



### 引用

- https://blog.csdn.net/weixin_42135399/article/details/101025941
- https://blog.csdn.net/lingzhou33/article/details/87901365
- https://blog.csdn.net/lingzhou33/article/details/87901365
