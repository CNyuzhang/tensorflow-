#  对应书本　[Tensorflow实战google深度学习框架](http://product.dangdang.com/24195829.html)


### 1、[conversion](https://github.com/CNyuzhang/tensorflow-/blob/master/mycode/conversation.py)
    介绍了tensorflow的基本使用，　包括简单计算，创建会话等

### ２、[second-classification](https://github.com/CNyuzhang/tensorflow-/blob/master/mycode/Second_classification.py)
在一个随机产生的数据集上进行神经网络的训练。总结出神经网络训练的三个步骤:
1. 定义神经网络的结构和前向传播的输出结果；
2. 定义损失函数以及选择反向传播优化算法；
3. 生成会话，并且在训练数据集上反复运行反向传播优化算法。

### ３、[Custom loss function](https://github.com/CNyuzhang/tensorflow-/blob/master/mycode/Custom%20loss%20function.py)
使用自定义的损失函数进行神经网络的训练

### 4、　[MNIST](https://github.com/CNyuzhang/tensorflow-/blob/master/mycode/MNIST%E5%85%A5%E9%97%A8/MNIST.py)
实现数字识别神经网络的训练。程序中涉及相关概念：
* tf.nn.   是tensorflow中的Neural Net　相关的。[nn的更多函数可见](https://www.tensorflow.org/api_docs/python/tf/nn)
* 滑动平均，又称指数加权平均，用来估计变量的局部均值。[更多内容](https://www.cnblogs.com/wuliytTaotao/p/9479958.html) 
* ReLU激活函数：常用几种激活函数<br/>
![](https://img-blog.csdn.net/20160706150849807)
* **softmax 函数**
    softmax是将神经网络得到的多个值，进行归一化处理，使得到的值在[0,1]之间，让结果变得可解释。即可以将结果看作是概率，某个类别概率越大，将样本归为该类别的可能性也就越高.
    ![](https://img-blog.csdn.net/20180914175343446?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0hlYXJ0aG91Z2Fu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 5、[mnist_with_variable_management](https://github.com/CNyuzhang/tensorflow-/blob/master/mycode/MNIST%E5%85%A5%E9%97%A8/mnist_with_variable_management.py)
将ＭＮＩＳＴ使用变量管理方法重构一遍。有关ｔｅｎｓｏｒｆｌｏｗ中变量管理的更多方法，参考[TensorFlow变量管理](https://www.jianshu.com/p/eedddcff65ff)
### 6、[tensorflow实现的LeNet-5模型](https://github.com/CNyuzhang/tensorflow-/tree/master/mycode/leNet5)
是第一個成功應用於數字識別問題的卷及神經網絡結構，一共有七层，两层卷积，两层池化，三层全连接层。
![LeNet-5](https://github.com/CNyuzhang/CNyuzhang.github.io/blob/master/img/cnn/LeNet_5.png?raw=true)
### 7、[Inception-v3模型](https://github.com/CNyuzhang/tensorflow-/blob/master/mycode/ML.py)
该模型与LeNet-5模型有较大的区别，在LeNet-5模型中，不同卷积层是通过串联的方式连接在一起，但是在Inception-v3模型中的Inception结构是将不同的卷积层通过并联的方式结合在一起．Inception-v3模型总共有４６层，由１１个inception模块构成
![inception-v3](https://github.com/CNyuzhang/CNyuzhang.github.io/blob/master/img/cnn/inception-v3.png?raw=true)

　　　　
　　　　
　　　


