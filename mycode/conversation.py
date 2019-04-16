#slove the problem of no compile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant([1.0, 2.0], name = "a")
b = tf.constant([2.0, 3.0], name = "b")

result = tf.add(a, b, name = "add")
print result

'''
sess = tf.Session()

#three way to output the value of tensor
with sess.as_default():
    print(result.eval())
print(sess.run(result))
print(result.eval(session=sess))
'''

#same as the above, but can simple the process of set the conversation as default
sess = tf.InteractiveSession()
print(result.eval())
sess.close()

#using configproto seting conversation
'''
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
print(result.eval())
'''
#print("\n the result is d% \n", result)
#print(a.graph is tf.get_default_graph())