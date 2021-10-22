## YOLOv3

YOLOv3的总体网络架构图
![](https://files.mdnice.com/user/15197/89af7410-6f8b-4f93-bb21-dfe57c360201.png)
可根据官方代码中的yolov3.cfg进行一一对应，标号$0$是第一个`[convolutional]`


**Darknet-53**

和DarkNet-19一样，同样下采样32倍。但是darknet-19是通过最大池化来进行，一共有5次。而darknet-53是通过尺寸2，步长为2的卷积核来进行的，也是5次。darknet-19是不存在残差结构(resblock，从resnet上借鉴过来)的，和VGG是同类型的backbone(属于上一代CNN结构)，而darknet-53是可以和resnet-152正面刚的backbone，并且FPS大幅提升，看下表：
![](https://files.mdnice.com/user/15197/ab2a82d9-0cf2-47ce-9aa4-3477c6510a3e.png)

**Neck层**

YOLO v3中的Neck层采用FPN(feature pyramid networks)的思想，会输出了3个不同尺度的特征，然后输入到YOLO Head头中进行预测。采用多尺度来对不同大小的目标进行检测。

* 在YOLO v3的总体架构图中可以看出，Neck层输出的特征图空间尺寸为$13 \times 13$是第81层；
* 然后它后退2层，然后将其2倍上采样。然后，YOLOv3将第61层网络输出的具有更高分辨率的特征图（尺寸为$26 \times 26$)，并使用concat将其与上采样特征图合并。YOLOv3在合并图上应用卷积滤波器以进行第二组预测
* 再次重复上一步骤，以使得到的特征图层具有良好的高级结构（语义）信息和目标位置的好的分辨率空间信息。

在YOLO v3中采用类似FPN的上采样和多尺度融合的做法（最后融合了3个尺度），在多个尺度的特征图上做检测，对于小目标的检测效果提升还是比较明显的。

**Head头**

为确定先验框priors，YOLOv3仍然应用k均值聚类。然后它预先选择9个聚类簇。对于COCO，锚定框的宽度和高度为$(10×13)，(16×30)，(33×23)，(30×61)，(62×45)，(59×119)，(116×90)，(156×198)，(373×326)$。这应该是按照输入图像的尺寸是$416×416$计算得到的。这9个priors根据它们的尺度分为3个不同的组。在检测目标时，给一个特定的特征图分配一个组。
![](https://files.mdnice.com/user/15197/5a0f9088-3d7b-4d23-95e0-fe7f6313a7cb.png)

YOLO v3输出了3个大小不同的特征图，从上到下分别对应深层、中层与浅层的特征。深层的特征图尺寸小，感受野大，有利于检测大尺度物体，而浅层的特征图则与之相反，更便于检测小尺度物体。每一个特征图上的一个点只需要预测3个先验框，YOLO v2中每个grid cell预测5个边界框，其实不然。因为YOLO v3采用了多尺度的特征融合，所以边界框的数量要比之前多很多，以输入图像为$416 \times 416$为例：
$$
13 \times 13 + 26 \times 26 + 52 \times 52)* 3 >> 13 \times 13 \times 5
$$
YOLO v3的先验框要比YOLO v2产生的框更多。

如果使用coco数据集，其有80个类别，因此一个先验框需要80维的类别预测值、4个位置预测及1个置信度预测，3个预测框一共需要3×(80+5)=255维，也就是每一个特征图的通道数 

**类别预测(Class Prediction)**

YOLO v3的另一个改进是使用了Logistic函数代替Softmax函数，以 处理类别的预测得分。原因在于，Softmax函数输出的多个类别预测之间会相互抑制，只能预测出一个类别，而Logistic分类器相互独立，可以实现多类别的预测。 实验证明，Softmax可以被多个独立的Logistic分类器取代，并且准确率不会下降，这样的设计可以实现物体的多标签分类，例如一个物体如果是Women时，同时也属于Person这个类别。 值得注意的是，Logistic类别预测方法在Mask RCNN中也被采用， 可以实现类别间的解耦。预测之后使用Binary的交叉熵函数可以进一步 求得类别损失。

**边界框预测和代价函数计算 (Bounding box prediction & cost function calculation)**

YOLOv3 使用逻辑回归Sigmoid预测每个边界框的置信度分数
YOLOv3改变了计算代价函数的方式。

* 如果边界框先验（锚定框）与GT目标的IOU比其他先验框大，则相应的目标性得分应为1。
* 对于重叠大于预定义阈值（默认值0.5）的其他先验框，不会产生任何代价。
* 每个GT目标仅与一个先验边界框相关联。 如果没有分配先验边界框，则不会导致分类和定位损失，只会有目标性的置信度损失。

> * 正样本：与GT的 IOU最大的框
> * 负样本：与GT的 IOU<0.5的框
> * 忽略的样本：与GT的 IOU>0.5但不是最大的值

一些YOLO的代码库都是使用配置文件配置网络结构，先提供一种使用pytorch实现的YOLOv3的code

* darknet53.py

```python
import sys

sys.path.append("..")

import torch.nn as nn
from model.layers.conv_module import Convolutional
from model.layers.blocks_module import Residual_block


class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        # 416*416*3 --> 416*416*32
        self.__conv = Convolutional(filters_in=3, filters_out=32, kernel_size=3, stride=1, pad=1, norm='bn',
                                    activate='leaky')
        # 416*416*32 -> 208*208*64
        self.__conv_5_0 = Convolutional(filters_in=32, filters_out=64, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 208*208*64 -> 208*208*32 -> 208*208*64
        self.__rb_5_0 = Residual_block(filters_in=64, filters_out=64, filters_medium=32)
        # 208*208*64 -> 104*104*128
        self.__conv_5_1 = Convolutional(filters_in=64, filters_out=128, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 104*104*128 -> 104*104*64 -> 104*104*128
        self.__rb_5_1_0 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)
        # 104*104*128 -> 104*104*64 -> 104*104*128
        self.__rb_5_1_1 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)
        # 104*104*128 -> 52*52*256
        self.__conv_5_2 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 52*52*256 -> 52*52*128 -> 52*52*256
        self.__rb_5_2_0 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_1 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_2 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_3 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_4 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_5 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_6 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_7 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        # 52*52*256 -> 26*26*512
        self.__conv_5_3 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 26*26*512 -> 26*26*256 -> 26*26*512
        self.__rb_5_3_0 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_1 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_2 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_3 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_4 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_5 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_6 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_7 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        # 26*26*512 -> 13*13*1024
        self.__conv_5_4 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        # 13*13*1024 -> 13*13*512 -> 13*13*1024
        self.__rb_5_4_0 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_1 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_2 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_3 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)

    def forward(self, x):
        x = self.__conv(x)

        x0_0 = self.__conv_5_0(x)
        x0_1 = self.__rb_5_0(x0_0)

        x1_0 = self.__conv_5_1(x0_1)
        x1_1 = self.__rb_5_1_0(x1_0)
        x1_2 = self.__rb_5_1_1(x1_1)

        x2_0 = self.__conv_5_2(x1_2)
        x2_1 = self.__rb_5_2_0(x2_0)
        x2_2 = self.__rb_5_2_1(x2_1)
        x2_3 = self.__rb_5_2_2(x2_2)
        x2_4 = self.__rb_5_2_3(x2_3)
        x2_5 = self.__rb_5_2_4(x2_4)
        x2_6 = self.__rb_5_2_5(x2_5)
        x2_7 = self.__rb_5_2_6(x2_6)
        x2_8 = self.__rb_5_2_7(x2_7)  # small

        x3_0 = self.__conv_5_3(x2_8)
        x3_1 = self.__rb_5_3_0(x3_0)
        x3_2 = self.__rb_5_3_1(x3_1)
        x3_3 = self.__rb_5_3_2(x3_2)
        x3_4 = self.__rb_5_3_3(x3_3)
        x3_5 = self.__rb_5_3_4(x3_4)
        x3_6 = self.__rb_5_3_5(x3_5)
        x3_7 = self.__rb_5_3_6(x3_6)
        x3_8 = self.__rb_5_3_7(x3_7)  # medium

        x4_0 = self.__conv_5_4(x3_8)
        x4_1 = self.__rb_5_4_0(x4_0)
        x4_2 = self.__rb_5_4_1(x4_1)
        x4_3 = self.__rb_5_4_2(x4_2)
        x4_4 = self.__rb_5_4_3(x4_3)  # large

        return x2_8, x3_8, x4_4

##### Test Code #####
# import torch
# from torchsummary import summary
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Darknet53().to(device=device)
# summary(model, (3, 416, 416))
```

* yolo_fpn

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.conv_module import Convolutional


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out


class FPN_YOLOV3(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """

    def __init__(self, fileters_in, fileters_out):
        super(FPN_YOLOV3, self).__init__()

        fi_0, fi_1, fi_2 = fileters_in
        fo_0, fo_1, fo_2 = fileters_out

        # large 输入：13*13*1024
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv0_0 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv0_1 = Convolutional(filters_in=1024, filters_out=fo_0, kernel_size=1, stride=1, pad=0)
        # 输出 13*13*3*(20+5)

        # 上采样准备与24*24*512的中等scale进行融合
        self.__conv0 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                                     activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route0 = Route()

        # medium 输入26*26*512
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1 + 256, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv1_0 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv1_1 = Convolutional(filters_in=512, filters_out=fo_1, kernel_size=1,
                                       stride=1, pad=0)
        # 输出 26*26*3*(20+5)

        # 上采样，准备与56*56*256的小scale进行融合
        self.__conv1 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                                     activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route1 = Route()

        # small
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2 + 128, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv2_0 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv2_1 = Convolutional(filters_in=256, filters_out=fo_2, kernel_size=1,
                                       stride=1, pad=0)
        # 输出 52*52*3*(20+5)

    def forward(self, x0, x1, x2):  # large, medium, small
        # large
        r0 = self.__conv_set_0(x0)  # DBL*5
        out0 = self.__conv0_0(r0)  # DBL
        out0 = self.__conv0_1(out0)  # conv -> 13*13*3*(20+5)

        # medium
        r1 = self.__conv0(r0)  # DBL
        r1 = self.__upsample0(r1)  # Upsample
        x1 = self.__route0(x1, r1)  # concat
        r1 = self.__conv_set_1(x1)  # DBL*5
        out1 = self.__conv1_0(r1)  # DBL
        out1 = self.__conv1_1(out1) # conv -> 26*26*3*(20+5)

        # small
        r2 = self.__conv1(r1)  # DBL
        r2 = self.__upsample1(r2)  # Upsample
        x2 = self.__route1(x2, r2)  # concat
        r2 = self.__conv_set_2(x2)  # DBL*5
        out2 = self.__conv2_0(r2)  # DBL
        out2 = self.__conv2_1(out2)  # conv -> 52*52*3*(20+5)

        return out2, out1, out0  # small, medium, large
```

* yolo_head

```python
import torch.nn as nn
import torch


class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors  # [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)]
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride  # 8

    def forward(self, p):  # p: [4, 75, 52, 52]
        # 获取batch_size 和 feature map的宽
        bs, nG = p.shape[0], p.shape[-1]

        # [batch_size, 3, (4,1,20), scale, scale] -> [batch_size, scale, scale, 3, (4,1,20)]
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)  # 4*52*52*3*25

        p_de = self.__decode(p.clone())

        return (p, p_de)

    def __decode(self, p):
        """ 解码过程
        1. 生成抛锚框
        2. 根据预测调整锚框
        """
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.__stride  # -8-/16/32
        anchors = (1.0 * self.__anchors).to(device)  # [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)]

        conv_raw_dxdy = p[:, :, :, :, 0:2]  # [batch_size, scale, scale, 3, 2]，获取最后一个维度中的前两维作为调整参数x,y
        conv_raw_dwdh = p[:, :, :, :, 2:4]  # [batch_size, scale, scale, 3, 2]，获取最后一个维度中的3和4维作为调整参数h,w
        conv_raw_conf = p[:, :, :, :, 4:5]  # [batch_size, scale, scale, 3, 1]，获取最后一个维度中的5维作为框内有无目标置信度
        conv_raw_prob = p[:, :, :, :, 5:]   # [batch_size, scale, scale, 3, 20]，获取最后一个维度中的后20维作为VOC数据类别的结果

        # !----------- 生成特征图的坐标点 ------------- #
        # [[0, ..., 0],[1, ..., 1],..., [51, ..., 51]]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size) # [52, 52]
        # [[0, ..., 51],[0, ..., 51],..., [0, ..., 51]]
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)  # [52, 52]
        grid_xy = torch.stack([x, y], dim=-1) # 相当于标记处特征图的每个格子左上角的坐标
        # 因为要生成3个先验框，所以也可以生成三个特征图，每个特征图对应不同的先验框
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        # 中心点调整参数进行Sigmoid操作归一化到[0,1]，然后对锚框的中心点进行调整
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride # grid_xy: 锚框的中心点，
        # 宽高调整参数进行指数运算，然后对锚框的宽高进行调整
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride  # anchors: 锚框的初始换宽高

        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)

        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox
```

* loss

```python
import sys

sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import tools
import config.yolov3_config_voc as cfg


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)

        return loss


class YoloV3Loss(nn.Module):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5):
        super(YoloV3Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides

    def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self.__strides

        loss_s, loss_s_giou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_sbbox,
                                                                                 sbboxes, strides[0])
        loss_m, loss_m_giou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
                                                                                 mbboxes, strides[1])
        loss_l, loss_l_giou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[2], p_d[2], label_lbbox,
                                                                                 lbboxes, strides[2])

        loss = loss_l + loss_m + loss_s
        loss_giou = loss_s_giou + loss_m_giou + loss_l_giou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls

        return loss, loss_giou, loss_conf, loss_cls

    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
        (1)The loss of regression of boxes.
          GIOU loss is defined in  https://arxiv.org/abs/1902.09630.

        Note: The loss factor is 2-w*h/(img_size**2), which is used to influence the
             balance of the loss value at different scales.
        (2)The loss of confidence.
            Includes confidence loss values for foreground and background.

        Note: The backgroud loss is calculated when the maximum iou of the box predicted
              by the feature point and all GTs is less than the threshold.
        (3)The loss of classes。
            The category loss is BCE, which is the binary value of each class.

        :param stride: The scale of the feature map relative to the original image

        :return: The average loss(loss_giou, loss_conf, loss_cls) of all batches of this detection layer.
        """
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid

        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        p_d_xywh = p_d[..., :4]

        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]

        # loss giou
        giou = tools.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)
        loss_giou = label_obj_mask * bbox_loss_scale * (1.0 - giou) * label_mix

        # loss confidence
        iou = tools.iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()

        loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) +
                     label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)) * label_mix

        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix

        loss_giou = (torch.sum(loss_giou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_giou + loss_conf + loss_cls

        return loss, loss_giou, loss_conf, loss_cls


if __name__ == "__main__":
    from model.yolov3 import Yolov3

    net = Yolov3()

    p, p_d = net(torch.rand(3, 3, 416, 416))
    label_sbbox = torch.rand(3, 52, 52, 3, 26)
    label_mbbox = torch.rand(3, 26, 26, 3, 26)
    label_lbbox = torch.rand(3, 13, 13, 3, 26)
    sbboxes = torch.rand(3, 150, 4)
    mbboxes = torch.rand(3, 150, 4)
    lbboxes = torch.rand(3, 150, 4)

    loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss(cfg.MODEL["ANCHORS"], cfg.MODEL["STRIDES"])(p, p_d, label_sbbox,
                                                                                                  label_mbbox,
                                                                                                  label_lbbox, sbboxes,
                                                                                                  mbboxes, lbboxes)
    print(loss)
```

