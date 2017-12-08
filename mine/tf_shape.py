# coding=utf-8

import tensorflow as tf
import numpy as np

x = tf.constant([[1, 2, 3], [4, 5, 6]]
y = [[1, 2, 3], [4, 5, 6]]
z = np.arange(24).reshape([2, 3, 4]))

sess = tf.Session()
# tf.shape()
x_shape = tf.shape(x)  # x_shape 是一个tensor
y_shape = tf.shape(y)  # <tf.Tensor 'Shape_2:0' shape=(2,) dtype=int32>
z_shape = tf.shape(z)  # <tf.Tensor 'Shape_5:0' shape=(3,) dtype=int32>
print sess.run(x_shape)  # 结果:[2 3]
print sess.run(y_shape)  # 结果:[2 3]
print sess.run(z_shape)  # 结果:[2 3 4]

# a.get_shape()
x_shape = x.get_shape()  # 返回的是TensorShape([Dimension(2), Dimension(3)]),不能使用 sess.run() 因为返回的不是tensor 或string,而是元组
x_shape = x.get_shape().as_list()  # 可以使用 as_list()得到具体的尺寸，x_shape=[2 3]
y_shape = y.get_shape()  # AttributeError: 'list' object has no attribute 'get_shape'
z_shape = z.get_shape()  # AttributeError: 'numpy.ndarray' object has no attribute 'get_shape'