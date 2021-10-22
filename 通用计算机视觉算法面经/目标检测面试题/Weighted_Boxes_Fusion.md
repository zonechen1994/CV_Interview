## 目标框加权融合-$Weighted$ $Boxes$ $Fusion$

### 简介

在之前的文档中我们有介绍 $NMS$ 的原理和代码，今天我们来看一个在多个模型进行目标检测中对于目标候选框筛选性能更好的一种算法——$Weighted$ $Boxes$ $Fusion$，简称 $WBF$。

简单来说，$WBF$ 算法首先将所有边界框按照置信度分数的递减顺序进行排序；然后生成另一个可能的框“融合”(组合)列表，并尝试检查这些融合是否与原始框匹配；最后使用一个公式来调整坐标和框列表中所有框的置信度分数。下面我们就来详细看一下 $WBF$ 的计算流程和代码。

### 理论部分

如下图 $1$ 所示是原文中给出的一个 $NMS$/$soft-NMS$ 和 $WBF$ 的原理示意图。$NMS$ 在原理上是删除不符合筛选规则的框来得到最后的候选框，**而 $WBF$ 是针对于集成学习来设计的，即对多个模型预测出来的预测框采用融合的方式获取最终一个候选框，这样的话就可以综合各个模型下的对于同一个物体预测出来的 $bounding$ $box$，从而获得更好的性能。**

![图 $1$](https://files.mdnice.com/user/15207/3297a2e9-38c6-44b2-ad8d-f12d879786f6.png)

$WBF$ 的详细算法步骤如下：

1、建立两个链表 $List$ $B$ 和 $List$ $C$，其中 $B$ 中储存的是多个模型预测出来的每一个 $bounding$ $box$，然后对 $B$ 中的 $bounding$ $box$ 进行降序排列得到 $C$。

2、建立空链表 $List$ $L$ 和 $List$ $F$，其中 $F$ 是用于储存 $L$ 中每一个 $cluster$ 融合后的 $bounding$ $box$，$L$ 中的每一个 $postion$ 都储存是一个 $bounding$ $box$ 的集合，称之为 $cluster$。

3、遍历循环 $B$ 中的 $bounding$ $box$ 在 $F$ 中寻找对应匹配的 $bounding$ $box$，匹配规则是根据两个框的 $IOU$ 值来的，在原论文中匹配的 $IOU$ 为 0.55。

4、在步骤 $3$ 中如果没有找到匹配的 $bounding$ $box$，则将这个框加到 $L$ 和 $F$ 的尾部。

5、在步骤 $3$ 中如果找到了匹配的 $bounding$ $box$，则将这个框加入到 $L$，加入的位置是该框在 $F$ 中匹配到这个框的 $Index$。$L$ 中每个位置可能有多个框，需要根据这多个框更新对应 $F$[$index$]的值。

6、匹配到框后，使用 $L$[$pos$]每一个 $cluster$ 中的框($T$ 个)重新计算在 $F$[$pos$]中的框的坐标和 $score$，其中 $score$ 和坐标的计算方式如下,$score$ 计算是取算数平均值得到的，而坐标值是通过框的置信度 $score$ 和坐标值相乘然后累加再除以 $score$ 的累加值得到，这样做可以使得具有较大置信度的框比较小置信度的框对融合后坐标的贡献值更大。

$$
C = \frac{\sum_{i=1}^{T}C_i}{T}
$$

$$
X_{1,2} = \frac{\sum_{i=1}^{T}C_i*X_{1,2_i}}{\sum_{i=1}^{T}C_i}
$$

$$
Y_{1,2}=\frac{\sum_{i=1}^{T}C_i*Y_{1,2_i}}{\sum_{i=1}^{T}C_i}
$$

7、当 $B$ 中所有的框都循环完后，对于 $F$ 中每个框的 $score$ 进行 $re-scale$，原因是因为如果一个 $cluster$ 中的框的数量太少的话，可能意味着若干模型中只有很少的模型预测到了这个框，因此是需要减少这种情况下对应框的置信度，作者给出了两种如下 $re-scale$ 的方式。作者在原论文中指出，这两种方式没有显著的差异，第一种 $re-scale$ 方式的性能会略好一点点。

$$
C = C * \frac{min(T, N)}{N}
$$

$$
C = C * \frac{T}{N}
$$

具体实现代码如下：

```python
# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import numpy as np


def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    for t in range(len(boxes)):
        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            b = [int(label), float(score) * weights[t], float(box_part[0]), float(box_part[1]), float(box_part[2]), float(box_part[3])]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse 
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers. 
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model 
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable  
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0 
    
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2). 
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

```

### 实验结论

原论文中作者也给出了多个模型使用 $NMS$ 和 $WBF$ 后的 $mAP$ 对比图，结果如下所示，从图表可以看出，使用了 $WBF$ 后，相较于使用 $NMS$/$soft-NMS$ 各个模型下的 $mAP$ 值均有不同程度的提升。

![](https://files.mdnice.com/user/15207/6d4302c6-189a-4b93-a058-4c3d16a0be29.png)

![](https://files.mdnice.com/user/15207/26bb2b23-3e81-4336-aee2-d2bb1316da5d.png)

![](https://files.mdnice.com/user/15207/c70e146a-073c-4d05-994e-730c4f528425.png)

![](https://files.mdnice.com/user/15207/4a5ac45b-c72c-4853-90b2-e3589bfc0aaf.png)

### 总结

本文分享了 $Weighted$ $Boxes$ $Fusion$ 这种预测框加权融合的方式，对多个模型预测出来的框进行融合，这种方式采用了集成学习的思想，综合各个模型下的对于同一个物体预测出来的 $bounding$ $box$，从而获得更好的性能。

### 引用

- https://arxiv.org/pdf/1910.13302.pdf
- https://github.com/ZFTurbo/Weighted-Boxes-Fusion
- https://blog.csdn.net/lyk_ffl/article/details/116024800
