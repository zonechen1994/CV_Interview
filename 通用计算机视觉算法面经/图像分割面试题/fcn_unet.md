## FCN与U-Net

### FCN
传统的CNN分类模型是在若干卷积层之后连上全连接层，将卷积层生成的特征图映射成一个固定长度（一般都是数据集的类别数）的特征向量。CNN分类模型的强大之处在于它的多层结构能够自动学习图像特征，并且可以学习到多个层次的特征：较浅的卷积层感受野较小，学习到的是一些局部区域的特征（如边缘、颜色、纹理等）；对于较深层的卷积层感受野较大，能够学习到更加抽象的特征，这些抽象的特征具有丰富的语义信息完成对目标物体的表示，但是其对物体的大小、位置等非常不敏感（不变性），这些丰富的语义信息有助于提高分类性能。

但是语义分割是一种像素级的分类，输出是与输入图像大小相同的分割图，输出图像的每个像素对应输入图像每个像素的类别，这时候需要物体的一些细节特征。

FCN将传统CNN中的全连接层转化为一个个的卷积层，如下图所示：
![所有的层都是卷积层，故称其为全卷积网络](https://user-images.githubusercontent.com/47493620/119691079-d8f3a380-be7c-11eb-98f8-7728487357d2.png)  

整体的网络结构分为两个部分：全卷积部分和反卷积部分。其中全卷积部分借用了一些经典的CNN网络（如AlexNet，VGG，GoogLeNet等），并把最后的全连接层换成卷积，用于提取特征，形成热点图；反卷积部分则是将小尺寸的热点图上采样得到原尺寸的语义分割图像。

![FCN结构图](https://files.mdnice.com/user/15197/3008c5d0-46c7-4233-9bb5-e6d1e5252604.png)

输入图像经过卷积和池化之后，得到的 feature map 宽高相对原图缩小了数倍，所产生图叫做heatmap热图，热图就是我们最重要的高维特诊图，得到高维特征的heatmap之后就是最重要的一步也是最后的一步对原图像进行upsampling，把图像进行放大、放大、放大，到原图像的大小。

![](https://files.mdnice.com/user/15197/c3b09780-c432-444a-ab23-161e157e9115.png)

最后的输出是（类别数）张heatmap经过upsampling变为原图大小的图片，为了对每个像素进行分类预测label成最后已经进行语义分割的图像，这里有一个小trick，就是最后通过逐个像素地求其在1000张图像该像素位置的最大数值描述（概率）作为该像素的分类。因此产生了一张已经分类好的图片，如下图右侧有狗狗和猫猫的图。

FCN上采样使用的是反卷积，也叫转置卷积，在以后的文章在做详细解释。

为了得到更好的分割效果，论文提出几种方式FCN-32s、FCN-16s、FCN-8s，如下图所示：
![](https://files.mdnice.com/user/15197/7fca5e06-0557-49f9-8536-50975d2ef6d3.png)

- 网络对原图像image进行卷积conv1、pool1后原图像缩小为1/2；
- 之后对图像进行第二次conv2、pool2后图像缩小为1/4；
- 接着继续对图像进行第三次卷积操作conv3、pool3缩小为原图像的1/8，此时保留pool3的featureMap；
- 接着继续对图像进行第四次卷积操作conv4、pool4，缩小为原图像的1/16，保留pool4的featureMap；
- 最后对图像进行第五次卷积操作conv5、pool5，缩小为原图像的1/32；然后把原来CNN操作中的全连接变成卷积操作conv6、conv7，图像的featureMap数量改变但是图像大小依然为原图的1/32，此时图像不再叫featureMap而是叫heatMap。

现在我们有1/32尺寸的heatMap，1/16尺寸的featureMap和1/8尺寸的featureMap，将1/32尺寸的heatMap进行upsampling操作到原始尺寸，这种模型叫做FCN-32s。这种模型暴力还原了conv5中的卷积核中的特征，一些细节是无法恢复的，限于精度问题不能够很好地还原图像当中的特征，精度很差。

所以自然而然的就想到将浅层网络提取的特征和深层特征相融合，能够更好的恢复细节信息。

把conv4中的卷积核对conv7 2倍上采样之后的特征图进行融合，然后这时候特征图的尺寸为原始图像的1/16，所以在上采样16倍，得到原始图像大小的特征图，这种模型叫做FCN-16s。

为了进一步恢复特征细节信息，把pool3后的特征图、对conv7上采样4倍的特征图和对pool4进行上采样2倍的特征图进行融合，此时的特征图尺寸为原始图像的1/8。融合之后在上采样8倍得到原始图像大小的特征图，这种模型叫做FCN-8s。

三种模型的分割效果图如下：
![](https://files.mdnice.com/user/15197/7629386f-9f27-4d35-a385-5021772b397e.png)

图中可以看出，FCN-8s的细节特征最为丰富，分割效果良好。同时论文中也尝试了将pool2、pool1的特征图进行融合，但是效果提升不明显，所以最终效果最好的就是FCN-8s。

FCN仍有一些缺点，比如：
- 得到的结果还不够精细，对细节不够敏感；
- 没有考虑像素与像素之间的关系，缺乏空间一致性等。

**代码实现**
1. BackBone is VGG, need to save the feature map of each pooling layers
```
class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()

        # conv1 1/2
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2 1/4
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3 1/8
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4 1/16
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5 1/32
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # load pretrained params from torchvision.models.vgg16(pretrained=True)
        if pretrained:
            pretrained_model = vgg16(pretrained=pretrained)
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.load_state_dict(new_dict)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)
        pool1 = x

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)
        pool2 = x

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)
        pool3 = x

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)
        pool4 = x

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)
        pool5 = x

        return pool1, pool2, pool3, pool4, pool5
```
2. FCN-8s
```
class FCNs(nn.Module):
    def __init__(self, num_classes, backbone="vgg"):
        super(FCNs, self).__init__()
        self.num_classes = num_classes
        if backbone == "vgg":
            self.features = VGG()

        # deconv1 1/16
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        # deconv1 1/8
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        # deconv1 1/4
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # deconv1 1/2
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # deconv1 1/1
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        y = self.bn1(self.relu1(self.deconv1(features[4])) + features[3])

        y = self.bn2(self.relu2(self.deconv2(y)) + features[2])

        y = self.bn3(self.relu3(self.deconv3(y)))

        y = self.bn4(self.relu4(self.deconv4(y)))

        y = self.bn5(self.relu5(self.deconv5(y)))

        y = self.classifier(y)

        return y
```

### U-Net

![U-Net网络结构图](https://files.mdnice.com/user/15197/48cef285-4132-489e-902b-6d5f3afd619b.png)

U-Net网络结构如上图所示。它由一个收缩路径（左侧）和一个扩展路径（右侧）组成。收缩路径遵循卷积网络的典型架构。它由两个3x3卷积（未填充卷积）的重复应用组成，每个卷积后跟一个整流线性单位（ReLU）和一个2x2最大池化操作，步长为2用于下采样。在每个降采样步骤中，我们将特征通道的数量增加一倍。

扩展路径中的每个步骤都包括对特征图进行上采样，然后进行2x2卷积（”上卷积”），以将特征通道的数量减半，并与从收缩路径中相应裁剪的特征图进行级联，再进行两个3x3卷积，每个卷积都由一个ReLU进行。由于每次卷积中都会丢失边界像素，因此有必要进行裁剪。在最后一层，使用1x1卷积将每个64分量特征向量映射到所需的类数。该网络总共有23个卷积层。

Unet相比更早提出的FCN网络，
- FCN是通过特征图对应像素值的相加来融合特征的；
- U-net使用拼接来作为特征图的融合方式，通过通道数的拼接，这样可以形成更厚的特征，当然这样会更佳消耗显存；

Unet的好处我感觉是：网络层越深得到的特征图，有着更大的视野域，浅层卷积关注纹理特征，深层网络关注本质的那种特征，所以深层浅层特征都是有各自的意义；另外一点是通过反卷积得到的更大的尺寸的特征图的边缘，是缺少信息的，毕竟每一次下采样提炼特征的同时，也必然会损失一些边缘特征，而失去的特征并不能从上采样中找回，因此通过特征的拼接，来实现边缘特征的一个找回。

```
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace

def contracting_block(in_channels, out_channels):
    """压缩路径的卷积单元
    """
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),  
    )
    return block

    
class expansive_block(nn.Module):
    """扩展路径
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()
        self.up_sample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels//2,
            kernel_size=(3, 3), 
            stride=2, 
            padding=1,
            output_padding=1
        )
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),  
        )
    
    def forward(self, e, d):
        """
        e: 压缩路径的特征图
        d: 当前的特征图
        """
        d = self.up_sample(d)
        # split & concat
        diffY = e.size()[2]-d.size()[2]  # (1, 3, `64`, 64)
        diffX = e.size()[3]-d.size()[3]  # (1, 3, 64, `64`)
        e = e[:,:, diffY//2:e.size()[2]-diffY//2, diffX//2:e.size()[3]-diffX//2]
        cat = torch.cat([e, d], dim=1)  # 按照通道拼接
        out = self.block(cat)
        return out
    

def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3,3), in_channels=in_channels, out_channels=out_channels),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block


# Bottleneck
def bottleneck(in_channels, out_channels):
    """瓶颈层
    """
    bottleneck = nn.Sequential(
        nn.Conv2d(kernel_size=(3,3), in_channels=in_channels, out_channels=out_channels),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(1024),
        nn.Conv2d(kernel_size=(3,3), in_channels=out_channels, out_channels=out_channels),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(1024)
    )
    return bottleneck
  

class UNet(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        # encoder
        self.encoder1 = contracting_block(in_channels=in_channel, out_channels=64)
        self.encoder1_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = contracting_block(in_channels=64, out_channels=128)
        self.encoder2_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = contracting_block(in_channels=128, out_channels=256)
        self.encoder3_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = contracting_block(in_channels=256, out_channels=512)
        self.encoder4_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        # Bottleneck
        self.bottleneck = bottleneck(in_channels=512, out_channels=1024)
        
        # decoder
        self.deconder4 = expansive_block(1024, 512, 512)
        self.deconder3 = expansive_block(512, 256, 256)
        self.deconder2 = expansive_block(256, 128, 128)
        self.deconder1 = expansive_block(128, 64, 64)
        
        # final layer
        self.final_layer = final_block(64, out_channel)
        
    def forward(self, x):
        # Encoder
        encoder_block1 = self.encoder1(x);print('encoder_block1:{}'.format(encoder_block1.size()))
        encoder_pool1 = self.encoder1_pool1(encoder_block1);print('encoder_pool1:{}'.format(encoder_block1.size()))
        encoder_block2 = self.encoder2(encoder_pool1);print('encoder_block2:{}'.format(encoder_block2.size()))
        encoder_pool2 = self.encoder2_pool2(encoder_block2);print('encoder_pool2:{}'.format(encoder_pool2.size()))
        encoder_block3 = self.encoder3(encoder_pool2);print('encoder_block3:{}'.format(encoder_block3.size()))
        encoder_pool3 = self.encoder3_pool3(encoder_block3);print('encoder_pool3:{}'.format(encoder_pool3.size()))
        encoder_block4 = self.encoder4(encoder_pool3);print('encoder_block4:{}'.format(encoder_block4.size()))
        encoder_pool4 = self.encoder4_pool4(encoder_block4);print('encoder_pool4:{}'.format(encoder_pool4.size()))

        # Bottleneck
        bottleneck = self.bottleneck(encoder_pool4);print('bottleneck:{}'.format(bottleneck.size()))
        
        # Decoder
        decoder_block4 = self.deconder4(encoder_block4, bottleneck);print('decoder_block4:{}'.format(decoder_block4.size()))
        decoder_block3 = self.deconder3(encoder_block3, decoder_block4);print('decoder_block3:{}'.format(decoder_block3.size()))
        decoder_block2 = self.deconder2(encoder_block2, decoder_block3);print('decoder_block2:{}'.format(decoder_block2.size()))
        decoder_block1 = self.deconder1(encoder_block1, decoder_block2);print('decoder_block1:{}'.format(decoder_block1.size()))
        
        # final layer
        final_layer = self.final_layer(decoder_block1)
        return final_layer
        

### 测试代码
image_ex = torch.rand((1, 3, 572, 572))
image_ex.size()
unet = UNet(in_channel=3, out_channel=1)
mask = unet(image_ex)
```
2. 输出
```shell
encoder_block1:torch.Size([1, 64, 568, 568])
encoder_pool1:torch.Size([1, 64, 568, 568])
encoder_block2:torch.Size([1, 128, 280, 280])
encoder_pool2:torch.Size([1, 128, 140, 140])
encoder_block3:torch.Size([1, 256, 136, 136])
encoder_pool3:torch.Size([1, 256, 68, 68])
encoder_block4:torch.Size([1, 512, 64, 64])
encoder_pool4:torch.Size([1, 512, 32, 32])
bottleneck:torch.Size([1, 1024, 28, 28])
decoder_block4:torch.Size([1, 512, 52, 52])
decoder_block3:torch.Size([1, 256, 100, 100])
decoder_block2:torch.Size([1, 128, 196, 196])
decoder_block1:torch.Size([1, 64, 388, 388])
```

**为什么Unet在医疗图像分割种表现好**

- 医疗影像语义较为简单、结构固定。因此语义信息相比自动驾驶等较为单一，因此并不需要去筛选过滤无用的信息。医疗影像的所有特征都很重要，因此低级特征和高级语义特征都很重要，所以U型结构的skip connection结构（特征拼接）更好派上用场

- 医学影像的数据较少，获取难度大，数据量可能只有几百甚至不到100，因此如果使用大型的网络例如DeepLabv3+等模型，很容易过拟合。大型网络的优点是更强的图像表述能力，而较为简单、数量少的医学影像并没有那么多的内容需要表述，因此也有人发现在小数量级中，分割的SOTA模型与轻量的Unet并没有什么优势

- 医学影像往往是多模态的。比方说ISLES脑梗竞赛中，官方提供了CBF，MTT，CBV等多中模态的数据（这一点听不懂也无妨）。因此医学影像任务中，往往需要自己设计网络去提取不同的模态特征，因此轻量结构简单的Unet可以有更大的操作空间。


现在医学图像领域，各种魔改unet，如VNet、UNet++、Attention-UNet、nnUNet等等。现在Transformer大火，又有一系列的基于Transformer-UNet的论文出现，下篇文章总结一下各种UNet的变种。

### 参考
1. https://www.jianshu.com/p/d6e3f21eb0b4
2. https://zhuanlan.zhihu.com/p/77201674
3. https://www.jianshu.com/p/d6e3f21eb0b4
4. https://blog.csdn.net/weixin_43143670/article/details/104791946
5. https://github.com/DarrenmondZhang/U_Net-DeepLabV3_Plus/blob/master/U-Net.ipynb
6. https://blog.csdn.net/Formlsl/article/details/80373200
7. https://zhuanlan.zhihu.com/p/57859749
 
