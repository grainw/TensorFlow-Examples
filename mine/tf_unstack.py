# coding=utf-8

import tensorflow as tf
import numpy as np

a = np.random.randn(5,5)
print a

# a = tf.random_normal([28, 28 ])
#
# print a
a = tf.constant(a)
b = tf.unstack(a , 28 , 1)

print b
with tf.Session() as sess:
    print (sess.run(a))

    print sess.run()

