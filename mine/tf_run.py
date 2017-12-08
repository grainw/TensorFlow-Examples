# coding=utf-8

import tensorflow as tf
import collections

a = tf.constant([10, 20])
b = tf.constant([1.0, 2.0])
# 'fetches' 可以为单个数
session = tf.Session()
v = session.run(a)
# v is the numpy array [10, 20]
# 'fetches' 可以为一个list.
v = session.run([a, b])
# v a Python list with 2 numpy arrays: the numpy array [10, 20] and the
# 1-D array [1.0, 2.0]
# 'fetches' 可以是 lists, tuples, namedtuple, dicts中的任意:
MyData = collections.namedtuple('MyData', ['a', 'b'])
v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
# v 为一个dict，并有
# v['k1'] is a MyData namedtuple with 'a' the numpy array [10, 20] and
# 'b' the numpy array [1.0, 2.0]
# v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
# [10, 20].
x = [1,2,3,4,5,6,7,8,9]
x = tf.reshape(x, shape=[-1, 3, 3, 1])