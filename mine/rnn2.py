import tensorflow as tf
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold  # StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
# from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
#% matplotlib inline

# hyperparameters
lr = 0.001
training_iters = 10000
batch_size = 200  # 3200/200=16用作训练 800/200用作测试

n_inputs = 35  # 输入35维的向量
n_steps = 203  # time steps
n_hidden_units = 100  # neurons in hidden layer
n_classes = 2

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

indices = label
depth = 2
on_value = 1
off_value = 0
output = tf.one_hot(indices, depth, on_value, off_value, axis=1)


def RNN3(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    # X(128 batch,28 steps,28 inputs)
    # ==>(128*28,28 inputs)
    X = tf.reshape(X, [-1, n_inputs])  # -1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1
    # ==>(128 batch*28 steps,128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # ==>(128 batch,28 steps,128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # cell
    ##########################################
    # state_is_tuple=True的时候，state是元组形式，state=(c,h)。如果是False，那么state是一个由c和h拼接起来的张量，state=tf.concat(1,[c,h])
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    dropout_lstm = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    # lstm cell is divided into two parts(c_state,m_state)
    _init_state = dropout_lstm.zero_state(batch_size, dtype=tf.float32)

    # choose rnn how to work,lstm just is one kind of rnn,use lstm_cell for active function,set initial_state
    outputs, final_state = tf.nn.dynamic_rnn(dropout_lstm, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    #     results = tf.matmul(final_state[1],weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (batch_size, n_classes)
    print(results.shape)
    return results


pred = RNN3(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))  # (pred,y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

predict_prob = tf.nn.softmax(pred)  # 得到对应预测标签的概率值
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # 返回true/false
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
with tf.Session() as sess:
    labelR = sess.run(output)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(label, n_folds=5)
    finalRes = []

    for numFold, (train_index, test_index) in enumerate(cv):
        sess.run(init)

        if len(train_index) < len(label) * 4 / 5:
            train_index = np.append(train_index, [0])
        if len(test_index) < len(label) / 5:
            np.append(test_index, [0])
            test_index = np.append(test_index, [0])
        if len(train_index) > len(label) * 4 / 5:
            train_index = train_index[0:len(label) * 4 / 5]
        if len(test_index) > len(label) / 5:
            test_index = test_index[0:len(label) / 5]
        x_train = [X[i] for i in train_index]

        y_train = [labelR[i] for i in train_index]
        x_test = [X[i] for i in test_index]
        y_test = [labelR[i] for i in test_index]
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        epoch = 0  # 统计迭代所有训练集的次数
        maxAccuracy = 0  # 连续5次不大于验证集最大准确性则 early stopping
        failNum = 0  # 统计连续不大于最大准确性的次数
        while epoch < training_iters:
            index = [i for i in range(len(x_train))]
            for step in range(int(len(x_train) / batch_size)):  # 每个batch的数据全部执行完
                indexR = random.sample(index, batch_size)  # batch_size=200
                batch_xs = np.array([x_train[i] for i in indexR])
                batch_ys = np.array([y_train[i] for i in indexR])

                batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
                # 从index中删除已训练的id，避免下次重复训练
                indexR = set(indexR)
                for ind in indexR:
                    index.remove(ind)
                sess.run([train_op], feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })

            if epoch % 30 == 0:  # 每30epoch输出此刻准确性
                accur = sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })
                print('%s%d%s%f' % ('At ', epoch, 'th accuracy:', accur))
                valiAccur = sess.run(accuracy, feed_dict={x: x_test[0:batch_size].reshape([-1, n_steps, n_inputs]),
                                                          y: y_test[0:batch_size]})  # 测试集中拿出一份用于验证集
                if valiAccur > maxAccuracy:
                    maxAccuracy = valiAccur
                    failNum = 0
                else:
                    failNum += 1
                costVal = sess.run(cost, feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })
                print('%s%f' % ('cost:', costVal))

            if failNum >= 5:
                print('%s%f' % ('Accuracy on validation set:', valiAccur))
                break
            epoch += 1

            # 测试
        #         x_test = np.array(x_test)
        #         y_test = np.array(y_test)
        x_test = x_test.reshape([-1, n_steps, n_inputs])
        result = []

        prob = []  # 保存最后预测每个label的概率
        final_label = []
        for i in range(1, 5):
            x_test2 = x_test[batch_size * (i - 1):batch_size * i]
            #             print(x_test2.shape)
            y_test2 = y_test[batch_size * (i - 1):batch_size * i]

            temp_prob = sess.run(predict_prob, feed_dict={x: x_test2, y: y_test2})

            temp_label = sess.run(tf.argmax(y_test2, 1))
            final_label.extend(temp_label)
            temp_prob2 = np.array(temp_prob)
            prob.extend(temp_prob2[:, 1])

            result.append(sess.run(accuracy, feed_dict={x: x_test2, y: y_test2}))

        fpr, tpr, thresholds = roc_curve(final_label, prob, pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.6f)' % (numFold, roc_auc))

        print('%d%s%f' % (numFold, "th fold accuracy：", np.mean(result)))
        finalRes.append(np.mean(result))
    print("Testing accuracy：", np.mean(finalRes))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线
    mean_tpr /= len(cv)  # 在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    # 画平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.6f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
