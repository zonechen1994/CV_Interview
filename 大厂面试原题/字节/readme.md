## 字节跳动面经题

-  了解anchor-free?

- anchor-based 和anchor-free区别

- 小目标有什么优化方面？输入分辨率，专门的一些网络(coco大小目标)，注意力机制，fpn

- 算法题：电梯没有带8的数字（提示：8进制和10进制的转换，此题是特殊的9进制）

- 算法题NMS

- yolo比RetinaNet的优势

- 旋转框检测的时候和水平框的区别

- 介绍半监督方法

- 常用的分类损失和常用的回归损失

- focal loss中的参数，哪个关注难样本，哪个解决长尾问题

-  如果label中有错误的标签，但我们却不知道，怎样解决（当时答的是多个模型投票，没有得到回应，开放题）

- anchor怎么设置，不同网络anchor设置的差别（SSD，faster-RCNN，yolo v3）

-  对IOU loss了解嘛？（CIOU，DIOU，GIOU）

- 对soft-nms了解嘛

-  nms的过程

- BN设置时是每个batch进行计算，但之前的计算资源比较匮乏，batch很小，所以BN计算的时候特别敏感，会震荡不能很好收敛解决办法？（每隔几个batch做BN，面试官同意了）