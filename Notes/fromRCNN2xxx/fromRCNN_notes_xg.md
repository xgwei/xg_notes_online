
# RCNN，fast， faster, YOLO v1,2,3, SSD

部分内容图片引用自下面的博客。其他废话是我的一点理解。

[深度学习目标检测：RCNN，Fast，Faster，YOLO，SSD比较](https://blog.csdn.net/ikerpeng/article/details/54316814)

[系统学习深度学习（三十二）--YOLO v1,v2,v3](https://blog.csdn.net/App_12062011/article/details/77554288)

**首先明确一点：所有这些都是想在目标classification的基础上加上定位功能。如果是别人训练好classification的任务，比如VGG19等，用classification train好的前半个提取特征的网络基本不用变！！！修改或者Fine tune 后半部分就好了**


![image](https://img-blog.csdn.net/20170110191535928?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaWtlcnBlbmc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

> 在Faster RCNN当中，一张大小为224*224的图片经过前面的5个卷积层，输出256张大小为13*13的 特征图（你也可以理解为一张13*13*256大小的特征图，256表示通道数）。接下来将其输入到RPN网络，输出可能存在目标的reign WHk个（其中WH是特征图的大小，k是anchor的个数）。
> 
> 实际上，这个RPN由两部分构成：一个卷积层，一对全连接层分别输出分类结果（cls layer）以及 坐标回归结果（reg layer）。卷积层：stride为1，卷积核大小为3*3，输出256张特征图（这一层实际参数为3*3*256*256）。相当于一个sliding window 探索输入特征图的每一个3*3的区域位置。当这个13*13*256特征图输入到RPN网络以后，通过卷积层得到13*13个 256特征图。也就是169个256维的特征向量，每一个对应一个3*3的区域位置，每一个位置提供9个anchor。于是，对于每一个256维的特征，经过一对 全连接网络（也可以是1*1的卷积核的卷积网络），一个输出 前景还是背景的输出2D；另一个输出回归的坐标信息（x,y,w, h,4*9D，但实际上是一个处理过的坐标位置）。于是，在这9个位置附近求到了一个真实的候选位置。


[看了这篇文章，了解深度卷积神经网络在目标检测中的进展](https://www.leiphone.com/news/201704/hbHyVvktQblDxyg0.html)

### R-CNN

![image](https://static.leiphone.com/uploads/new/article/740_740/201704/58fd6e59ce82d.png?imageMogr2/format/jpg/quality/90)



[Girshick, Ross, et al. “Rich feature hierarchies for accurate object detection and semantic segmentation.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.](https://arxiv.org/pdf/1311.2524v5.pdf)

1. 生成候选区域：使用了Selective Search从原图生成2000-3000个候选区域。（理论上实际可以使用任意算法生成这些区域。其实重点是想尽可能多的把可能的框都选出来。）
2. 每个ROI的图都resize成227x227输入网络。
3. 输出是21维的类别标号，表示20类+背景。（意思是这个框的内容可能是20类物体里面任意一个，也可能都不是，是背景（跟ground truth 重叠小过0.5的为背景）。）loss就变成了21类的error。

**以上为网络训练部分，可以想象这一步后已经能有结果了。输入2000个框，可能只有一部分被判别为20类的其中之一，但这样可能边框不准确，并且重叠的很多框都可能被标定出来。**

4. RCNN在这又加了个类别判断，前面不是已经有分类结果了吗，不知为何？（这样更准些？）当然原理不难，使用了个SVM做了个二分类one-vs-all。输入是网络最后输出的特征4096维。（这里正样本是overlap大于0.3的）
5. Bounding Box regressor。缩放平移=f(4096维特征)，用了个全连接网络。loss = ||gt缩放平移-f(4096维特征)||+lambda*||f(4096维特征)||,lambda用的10,000


==很慢，问题所在：==
- 一开始就选ROI，后面算特征会重复计算的。
- svm，bbox regressor单独训练，不够优雅

### Fast R-CNN

![image](https://static.leiphone.com/uploads/new/article/740_740/201704/58fd6e9104151.png?imageMogr2/format/jpg/quality/90)

1. 图像输入网络得到特征图矩阵，例如MxNx512大小，在这里选ROI。（不同于R-CNN先采样，这个后采样减少重复计算。用的也是selective search生成ROI）
2. 每个ROI的特征矩阵在这resize（用的maxpooling）成7x7x512。
3. 这些feature当然可以直接拿来做预测。但在这，前面feature不变，后面变成预测+bounding box regression一起做（feature跟R-CNN一样，当然可以拿来做BBox）。Loss函数一起训练，目标就是把所有东西都塞到网络里。loss如下，第一部分是预测偏差，第二部分是bbox的error

![image](https://static.leiphone.com/uploads/new/article/740_740/201704/58fd6eaa01c1e.png?imageMogr2/format/jpg/quality/90)

==问题：==
- 还是选ROI，这一步虽然在feature生成后才做，但用的也是selective search生成ROI,还是慢

### Faster R-CNN

这个图画的没上面的好
![image](https://static.leiphone.com/uploads/new/article/740_740/201704/58fd6e59da6c4.png?imageMogr2/format/jpg/quality/90)

可以想象这一步主要是想把选ROI也融入网络（用卷积神经网络生成ROI），从硬选变成回归问题。所以核心在region proposal network

1. 还是正图过模型中的算特征部分。
2. 这里做用个3x3的slidingwindow把这特征图遍历一遍，每个3x3里maxpooling做为结果。相当于把整图用grid划分，每个小格都判断是不是物体。这还不算完，每个点还有k个anchors（通常是9个）。这样每个点都形成了上图的回归问题：判断这个点的k种anchor是不是foreground+判断这个boundingbox的移动及缩放？（其实也是选了很多ROI啊，只不过每点都有个roi，所以可以网络做了，规则话就意味着可导了）

**上面做完基本就有了哪个点的哪个anchor是前景，平移缩放多少。**（会不会重复的很多？？后面怎么了？？）\
果然，proposal layer 里面就可以[做到](https://blog.csdn.net/weixin_35653315/article/details/54577675)
> 2. 利用NMS, non-maximum suppression在重叠的bbox筛选出得分最高的bbox. 
> 3. 其他筛选.

这是每个点对应的anchor的样子，实际由于这一步feature是原图缩小很多倍后的，每个3x3已经可以代表很大一片区域了。
[link](http://lib.csdn.net/article/deeplearning/61641)
> 注：关于上面的anchors size，其实是根据检测图像设置的。在python demo中，会把任意大小的输入图像reshape成800x600（即上文中提到的M=800，N=600）。再回头来看anchors的大小，anchors中长宽1:2中最大为352x704，长宽2:1中最大736x384，基本是cover了800x600的各个尺度和形状。
![image](http://img.blog.csdn.net/20170322103823615)

### YOLO

![image](https://img-blog.csdn.net/20171208114319192?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

> 改革了区域建议框式检测框架: RCNN系列均需要生成建议框，在建议框上进行分类与回归，但建议框之间有重叠，这会带来很多重复工作。YOLO将全图划分为SXS的格子，每个格子负责中心在该格子的目标检测，采用一次性预测所有格子所含目标的bbox、定位置信度以及所有类别概率向量来将问题一次性解决(one-shot)。

“~~again,前面的feature网络不变，最后一层feature~~ 原图划分成7x7的格子，每个格子直接regress 类别+boundingbox” 这是论文和各个论坛里的解释，实在看不明白是如何实现的

读代码的理解：
1. 固定图片大小，yolov2 为416
2. 一路卷积+batchnormalization+leakyReLU+MaxPooling（2，2）,大小416->208->104->52->26->13。最后为13x13xfeaturedim的特征tensor，每条feature都是原图13x13格子里的信息 （与之前设想相同，每个点都是特征的histogram了，）
3. 每个点都输出 5个box，每个box有（偏移缩放）4+（confi）1+class（80）,如下面代码，（**直接conv就行，这里虽然没有明确的物理意义及各个与测量的constraint，但输入的维度一致只要后期定义好loss就能regress。yolo v2 里，这5个box是不是直接信息，而是5个anchor的变换信息**）

```python
# Layer 22
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)
____________________________________________________________________________________________________
reshape_1 (Reshape)              (None, 13, 13, 5, 85) 0           conv_23[0][0]                    
____________________________________________________________________________________________________
```

要注意，groundtruth是50个框的buffer，如下，也作为一个输入，跟上面的输出再组合成loss。（==？？？class哪儿去了？？==）

```python
input_2 (InputLayer)             (None, 1, 1, 1, 50, 4 0                                    
```


（==有个疑惑，如果只用一个小格预测，比如上面的狗头，真的可以预测一个好的boundingbox吗，毕竟相对于整个狗，这个格子比例很小啊？？== 其实这个问题在convolution部分已经考虑到了，每个点看似代表原图13x13的一个格子，实际通过卷积把旁边的很多信息都卷到当前点了。）

region proposal能不能全部用convolution做呢？直接从原来的网络剪切过来，

最后一层，每个点做一次预测，就像上图下面那个gridmap。


如果从一开始就一路卷积下去呢？每一层都认为有效，直到图像卷成1x1x4096的向量（这时代表全图都是一个物体）

### SSD
> 我个人认为SSD可以理解为multi-scale版本的RPN，它和RPN最大的不同在于RPN只是在最后的feature map上预测检测的结果，而最后一层的feature map往往都比较抽象，对于小物体不能很好地表达特征，而SSD允许从CNN各个level的feature map预测检测结果，这样就能很好地适应不同scale的物体，对于小物体可以由更底层的feature map做预测。这就是SSD和RPN最大的不同，其他地方几乎一样。下图是SSD的网络结构，可以看到不同层的feature map都可以做预测。

![image](http://lufo.me/media/files/2016/10/04.jpg)
