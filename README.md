# master-cudnn
## Overview

NVIDIA cuDNN为深度学习中频繁使用的操作提供了高度调优的实现：

* 前向、反向卷积，以及互相关。
* 矩阵乘
* 前向、反向池化
* 前向、反向Softmax
* 前向、反向激活：`relu`，`tanh`，`sigmoid`，`elu`，`gelu`，`softplus`，`swish`
* 算术、数学、关系和逻辑关系的逐点运算（包括各种类型的前向、反向激活）
* 张量转换函数
* 前向、反向LRN，LCN，batch normalization，instance normalization，以及layer normalization

cuDNN不仅提供单个op的高性能实现，还支持一系列灵活的多op融合模式，用于进一步优化。cuDNN库的目标是在NVIDIA GPUs上为重要的深度学习用例提供最佳性能。

在cuDNN 7及之前的版本，各深度学习op以及融合模式被设计为一组固定的API，称为"legacy API"。从cuDNN 8开始，为了支持对流行的融合模式进行快速扩展，新增了"Graph API"，这些API允许用户通过定义计算图来表达计算，而不是通过一组固定的API调用来选择计算。这比"leagcy API"提供了更好的灵活性，对于大多数用例，现在推荐使用"Graph API"。

注意，cuDNN库同时提供了C API和一个开源的包裹C API的C++层，C++层可能对大部分用户来说更加方便。但是，C++层只支持"Graph API"，不支持"legacy API"。

