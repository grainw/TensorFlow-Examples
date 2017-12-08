# coding=utf-8

import tensorflow as tf

# case 2
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([1, 1, 5, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
res = (sess.run(op))
print (res.shape)

import tensorflow as tf

input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
res = (sess.run(op))
print (res.shape)


# 1、使用VALID方式,feature map的尺寸为
# out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))=(5-3+1)/1 = 3
#
# out_width = ceil(float(in_width - filter_width + 1) / float(strides[2])) = (5-3+1)/1 = 3
#
# 所以,feature map的尺寸为3*3
#
# 2、filter的参数个数为3*3*5*1,也即对于输入的每个通道数都对应于一个3*3的滤波器,然后共5个通道数,conv2d的过程就是对5个输入进行点击然后求和,
# 得到一张feature map。如果要得到3张feature map,那么应该使用的参数为3*3*5*3个参数.

import tensorflow as tf

a = tf.constant([
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0],
     [8.0, 7.0, 6.0, 5.0],
     [4.0, 3.0, 2.0, 1.0]],
    [[4.0, 3.0, 2.0, 1.0],
     [8.0, 7.0, 6.0, 5.0],
     [1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]]
])

a = tf.reshape(a, [1, 4, 4, 2])

pooling = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
with tf.Session() as sess:
    print("image:")
    image = sess.run(a)
    print (image)
    print("reslut:")
    result = sess.run(pooling)
    print (result)