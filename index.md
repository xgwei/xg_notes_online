
问题： 有两张图，想把一张的转换为另一张的风格

`$\mathbf{I}_{base}$`为内容图像， `$\mathbf{I}_{style}$`为风格图像, `$\mathbf{I}_{?}$`为合成的图像

现在想要的结果是`$\mathbf{I}_{?}$`跟`$\mathbf{I}_{base}$`在内容上相似，`$\mathbf{I}_{?}$`跟`$\mathbf{I}_{style}$`在风格上相似。

可以写出个目标函数了：

```math
E=\left \| f_{content}(\mathbf{I}_{?})-f_{content}(\mathbf{I}_{base}) \right \|_{anyerror}+\left \| f_{style}(\mathbf{I}_{?})-f_{style}(\mathbf{I}_{style}) \right \|_{anyerror}+TV(\mathbf{I}_{?})
```
content loss要求形状相似，style loss要求styple相似，TV loss 只是想让噪声小点看不出破绽
各个error之间当然要有个常数来确定下各自的权重。 有了这个函数，只要能求导，就能解出来了。

==那么问题来了，content loss， style loss 该怎么定义呢？==

### content loss
先上代码

- content loss function
```
def content_loss(base, combination):
    return K.sum(K.square(combination - base))
``` 
- content loss 用的层的结果
``` 
block5_conv2
``` 
content loss要求形状相似，所以用包含位置的square error
content只需要比较深的一层（downsample好多次后的特征）。这是可以想象的，因为可以想象风格转换后的图像内容不能跟原图太像，只要一个大致的样子相似。

### style loss

先上代码

```
# the gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
```
这里用到了5层的feature去作为风格相似度参照。为什么呢？可以想如果只有一层，feature dot product相似的意思就是两个加起来和相似，其实限制不了什么，仅仅改变scale就能做得到。如果多层feature dot 都相似，就有多一些的限制力在里面了。（不过只是每层和相似就能限制风格因素也挺令人惊讶的。）

    block1_conv1
    block2_conv1
    block3_conv1
    block4_conv1
    block5_conv1
### TV loss
最后当然再加个TV loss 让整个结果再平滑些，没有细小的噪声。这也是为什么通常用作风格化转换都是把真实图转换为油画或动画风格，因为TV只能消除细节并不能增加细节。可以想象把油画变真实应该效果不会太好。

### 一些思考

其实可以看到cnn提取的特征即被用作了content也被用作了style。
- content只需要比较深的一层（downsample好多次后的特征）。这是可以想象的，因为可以想象风格转换后的图像内容不能跟原图太像，只要一个大致的样子相似。同时content loss 如前所述是包含位置关系的square error，因此内容得以保留。
- style则需要多层限制，原因如前style loss里所述。


### vgg 19 model.summary()
这里再附上 vgg 19 的模型总结，上面提到的层都能在此找到

```
# vgg 19 model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_43 (InputLayer)        (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
_________________________________________________________________
```
