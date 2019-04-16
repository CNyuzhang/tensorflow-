#slove the problem of no compile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from numpy.random import RandomState

#定义训练数据大小
batch_size = 8

#定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

#定义神经网络前像传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(
    y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)

#通过随机数产生一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

#定义规则给出样本的标签，所有x1+x2<1样本被认为是正样本
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

#创建一个会话运行tenorflow程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    #初始化变量
    sess.run(init_op)
    #输出初始化的权重值
    #print sess.run(w1)
    #print sess.run(w2)

    #设定训练轮数
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end   = min(start+batch_size, dataset_size)

        #通过选取的样本训练神经网络病更新参数
        sess.run(train_step,
                feed_dict={x:X[start:end], y_: Y[start:end]})
        if i %1000 == 0:
            total_cross_entropy = sess.run(
                    cross_entropy, feed_dict={x:X, y_:Y})
            print("Affter %d training step(s), cross entropy on all data is %g" %
                            (i, total_cross_entropy))
    
    print sess.run(w1)
    print sess.run(w2)

