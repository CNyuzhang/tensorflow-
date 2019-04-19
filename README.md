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


　　　　
　　　　
　　　


