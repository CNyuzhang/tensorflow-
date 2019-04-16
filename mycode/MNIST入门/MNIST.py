import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import tensorflow as tf

#占位符placeholder，使用二维的浮点数张量表示图，张量的形状是【None,784], 表示第一个唯独可以任意长度
x = tf.placeholder("float", [None, 784])

#Variable表示可以修改的placeholder
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#实现softmax模型  tf.matmul(​​X，W)表示x乘以W
y = tf.nn.softmax(tf.matmul(x,w)+b)

#添加一个新的占位符用于输入正确值  计算交叉熵   tf.log 计算 y 的每个元素的对数
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#设置训练  梯度下降法 学习率0.01  
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化上面创建的变量
init = tf.initialize_all_variables()

#在一个session里面启动模型，初始化变量
sess = tf.Session()
sess.run(init)

#开始训练，循环训练1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)              #选择100个批处理数据点
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})   #使用这100个进行训练

#tf.argmax给出某个tensor对象在某一维上的其数据最大值所在的索引值
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



