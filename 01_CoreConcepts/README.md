# 核心概念

在讨论graph和legacy API的细节之前，这一部分先介绍两者共同的核心概念。

## cuDNN Handle

cuDNN库暴露了一系列主机API，但是假设所有用到GPU的op，其必要的数据都可以直接从设备上访问。

使用cuDNN的应用必须先调用`cudnnCreate()`来初始化句柄。这个句柄会被显示传递给后续操作GPU数据的函数。应用可以通过调用`cudnnDestroy()`来释放句柄关联的资源。这中方法允许用户在使用多个主机线程、多个GPU或多个CUDA流时显示控制cuDNN库的功能。

例如，一个应用可以使用`cudaSetDevice`（在创建cuDNN句柄之前）在不同的主机线程里关联不同的设备，然后再每个主机线程里，创建一个唯一的cuDNN句柄，用于指定该线程中所有的cuDNN API调用都作用于该线程关联的设备。这种情况下，使用不同句柄的cuDNN库调用将自动运行在不同的设备上。

cuDNN假定在`cudnnCreate()`创建句柄后，直到调用`cudnnDestroy()`销毁句柄之前，句柄所关联的设备保持不变。如果要在同一个主机线程中使用不同的设备，应用必须通过`cudaSetDevice`设定新设备，然后重新使用`cudnnCreate()`创建一个cuDNN句柄，这个句柄会关联到新设备。

## Tensors and Layouts

无论使用graph API还是legacy API，cuDNN操作都会使用tensors作为输入，并产生tensors作为输出。

### Tensor Descriptor

cuDNN库使用一个通用的n-D Tensor描述符来描述数据。可以通过如下参数来定义一个Tensor描述符：

* 数据的维度（3-8维）
* 数据类型（32-bit 浮点，64-bit 浮点，16-bit 浮点......）
* 一个定义每个维度值的整型数组
* 一个定义每个维度跨距的整型数组（例如：获取当前维度下一个元素所需要跨越的元素数）

这种tensor定义允许某些维度数据相互重叠，例如，当前维度的跨距 < 下一个维度值 * 下一个维度的跨距（举例：nchw = [1, 2, 3, 4]，假设w的跨距是1，当h的跨距为w，也就是4 * 1，就没有重叠，但是当h的跨距是2，也就是2 < 4 * 1，h维度和w维度就存在重叠）。**在cuDNN里，除非另外指定，所有执行前向传播的函数中，输入tensor允许overlapping，所有输出tensor不支持overlapping。虽然Tensor描述支持负跨距（对数据镜像很有用），但除非另有规定，否则cuDNN函数不支持具有负跨距的Tensor。**

#### WXYZ Tensor Descriptor

tensor的格式可以使用各个维度的缩写来标识（例如：NHWC），在cuDNN文档中，使用这种描述方法意味着：

* 所有的跨距都是正的
* 字母对应维度的跨距按降序排列（例如：NHWC排列的tensor，n_stride >= h_stride >= w_stride >= c_stride）

#### 3-D Tensor Descriptor

一个3-D tensor通常用于矩阵乘法。具有三个维度：B，M，N。B代表batch size（对于单batch的GEMM，设置为1），M和N分别代表矩阵的行列数。更多信息可以参考`CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR`op（cuDNN graph API）。

#### 4-D Tensor Descriptor

4-D tensor descriptor使用4个字母（N，C，H，W）表示多batch的2D图像，N，C，H，W分别代表batch size，特征图数量，图像高度，图像宽度。以跨距值来降序排列这4个字母。最常用的4-D tensor格式是：

* NCHW
* NHWC
* CHWN

#### 5-D Tensor Descriptor

5-D tensor descriptor使用5个字母（N，C，D，H，W）表示多batch的3D图像，N，C，D，H，W分别代表batch size，特征图数量，图像深度，图像高度，图像宽度。以跨距值来降序排列这5个字母。最常用的5-D tensor格式是：

* NCDHW
* NDHWC
* CDHWN

#### Fully-Packed Tensors

一个tensor只有满足如下条件时，才被定义为`XYZ-fully-packed`：

* tensor的维度数等于`fully-packed`前面的字母数（例如：`NCHW-full-packed`，则此tensor的维度数必须为4。）
* 第`i`维度的跨距必须等于第`i+1`维度值乘以第`i+1`维度的跨距
* 最后一个维度的跨距必须是1

例如：一个格式是`NHWC`的tensor，假设N，H，W，C分别是1，2，3，4，则根据第三条，最后一个维度`C`的跨距必须是1，然后循环使用第2条，则可以得到`W`的跨距必须是`1 * 4(C维度的值) = 4`，`H`的跨距必须是`3(W维度值) * 4(W的跨距) = 12`，`N`的跨距必须是`2(H维度值) * 12(H的跨距) = 24`。同时，根据第一条，此tensor的维度数是4（只包含NHWC 4个维度），则这个tensor才是`NHWC-full-packed`。

#### Partially-Packed Tensors

一个 WXYZ tensor只有满足如下条件时，才被定义为`XYZ-packed`：

* 不在`-packed`中的维度的跨距必须大于或等于下一个维度值和下一个维度跨距的乘积。
* 在`-packed`中的维度，第`i`维度的跨距必须等于第`i+1`维度值和第`i+1`维度跨距的乘积。
* 如果最后一个维度在`-packed`中，则它的跨距必须是1。

例如：一个4-D tensor的格式是`NCHW`，N，C，H，W分别是1，2，3，4，若其是`CH-packed`，则根据第三条，最后一个维度`W`不在`CH-packed`中，其跨距可以不为1，假设其跨距是2，同时，假设`H`的跨距是3（随意设置的值，为了证明`CH`是否packed与`W`维度无关，故意没有设置为`4(W维度值) * 2(W维度跨距) = 8`），则根据第二条，`H`在`CH-packed`中，则`C`的跨距必须等于`3(H维度值) * 3(H维度跨距) = 9`，根据第一条，`N`的跨距必须大于等于`2(C维度值) * 9(C维度跨距) = 18`，这样才能说这个tensor是`CH-packed`。

#### Spatially Packed Tensors

Spatially-packed tensors被定义为在空间维度上packed的partially-packed tensor。也就是说对于spatially-packed 4D tensor，意味着无论是NCHW tensor还是CNHW tensor，都是HW-packed。

#### Overlapping Tensors

如果在整个维度范围内迭代时，多次取到相同的地址，则该tensor被定义为overlapping tensor。实践中，overlapping tensors意味着，对于某些维度`i`（在`[1, nbDims]`区间内），会存在`stride[i-1] < stride[i] * dim[i]`。