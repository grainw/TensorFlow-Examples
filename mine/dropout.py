# coding=utf-8
import tensorflow as tf
sess = tf.InteractiveSession()
a = tf.get_variable('a',shape=[2,5])
b = a
a_drop = tf.nn.dropout(a,0.8)
sess.run(tf.initialize_all_variables())
print(sess.run(b))
#[[ 0.28667903 -0.66874665 -1.14635754  0.88610041 -0.55590457]
# [-0.29704338 -0.01958954  0.80359757  0.75945008  0.74934876]]
print(sess.run(a_drop))
#[[ 0.35834879 -0.83593333 -1.43294692  1.10762548 -0.        ]
# [-0.37130421 -0.          0.          0.94931257  0.93668592]]