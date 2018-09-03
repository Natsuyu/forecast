#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf
import pickle
import math
import codecs
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
import matplotlib.pyplot as plt
import numpy as np
# import config
from sklearn.externals import joblib

window_size = 5
embedding_size = 128
label_size = 5
unit_num = 128
layer_size = 2
id = 100 #id of word2vect
learning_rate = 0.01 #0.1 is not suitable
max_stc_size = 50
train_batch_size = config.train_batch_size
test_batch_size = 1
prob = 0.4
#embedding_size和unit_num要一样

labels = pickle.load(open("labels.txt",'r'))
word2id = pickle.load(open("word2id.txt",'r'))
id2word = pickle.load(open("id2word.txt",'r'))
s2id = pickle.load(open("s2id.txt",'r'))
test_label = pickle.load(open("test_label.txt",'r'))
test_id = pickle.load(open("test_ids.txt",'r'))
vocab_size = len(word2id.keys())+1

is_train = tf.placeholder(tf.int32) #true -> train


train_y = labels
train_x = s2id

test_y = test_label
test_x = test_id

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.batch(train_batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_dataset = test_dataset.batch(train_batch_size)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

train_initializer = iterator.make_initializer(train_dataset)
test_initializer = iterator.make_initializer(test_dataset)

# x_, y_ = iterator.get_next()

### 测试集的测试要加进去，但是iterator这个要注意不要重复

embed = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), tf.float32)
# ids = tf.placeholder(tf.int32, [None, max_stc_size])

x = tf.placeholder(tf.float32, [None, embedding_size*window_size])
# y = tf.placeholder(tf.int32, [None, max_stc_size])
label = tf.placeholder(tf.int32, [None])

ids, y = iterator.get_next()

#========= BiLSTM ==============
def lstm_cell(num_unit, prob=0.5):
    cell = tf.nn.rnn_cell.LSTMCell(num_unit, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-prob)

cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(unit_num, prob)] * layer_size)
cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(unit_num, prob)] * layer_size)
# print cell_fw,cell_bw
inputs_embed = tf.nn.embedding_lookup(embed, ids)

inputs_embed = tf.unstack(inputs_embed, max_stc_size, 1)

outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs_embed,dtype=tf.float32)

outputs = tf.stack(outputs, axis=1)
outputs = tf.reshape(outputs, [-1, unit_num * 2])
weight = tf.Variable(tf.random_normal([layer_size*unit_num, label_size]))
bias = tf.Variable(tf.random_normal([label_size]))
yy = tf.matmul(outputs, weight)+bias

#=========CRF==============
original_sequence_lengths = tf.fill([train_batch_size],max_stc_size)
scores = tf.reshape(yy, [-1, max_stc_size, label_size])
logits, trans_matrix = crf.crf_log_likelihood(scores, y, original_sequence_lengths)
loss = tf.reduce_mean(-logits)
viterbi_sequence, viterbi_score = crf.crf_decode(scores, trans_matrix, original_sequence_lengths)
init_trans = tf.Variable([label_size])

viterbi_sequence = tf.reshape(viterbi_sequence, [-1])

#========END CRF===========
# print viterbi_sequence
y_predict = tf.cast(tf.argmax(viterbi_sequence, axis=1), tf.int32)
# Reshape y_label
y_label_reshape = tf.cast(tf.reshape(y, [-1]), tf.int32)
# Prediction
y_viterbi = tf.placeholder(tf.int32, [None])
correct_prediction1 = tf.equal(y_predict, y_label_reshape)
correct_prediction = tf.equal(y_viterbi, y_label_reshape)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
# Loss
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape, logits=tf.cast(yy, tf.float32)))
# Train
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def viterbi(yy,A,initA):
    rows = train_batch_size * max_stc_size

    path = np.ones([label_size, rows], dtype=np.int32) * -1
    corr_path = np.zeros([rows], dtype=np.int32)
    scores = np.zeros([label_size, rows], dtype=np.float64) * (np.finfo('f').min / 2)
    # print scores[:, 0], initA, yy[0,:]
    scores[:, 0] = (yy[0,:]+initA)
    label = ['x', 's', 'b', 'm', 'e']

    for pos in range(1, rows):
        for t in range(label_size):
            for prev in range(label_size):
                    temp = scores[prev][pos - 1] + A[prev][t] + yy[pos][t]
                    if temp > scores[t][pos]:
                        path[t][pos] = prev
                        scores[t][pos] = temp

    max_index = np.argmax(scores[:, -1])
    corr_path[rows - 1] = max_index
    for i in range(rows - 1, 0, -1):
      max_index = path[max_index][i]
      corr_path[i - 1] = max_index

    return scores, corr_path

ax =[]
ay = []
bx = []
by = []
ctrain = []
ctest = []
step = 500

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(train_initializer)
    # sess.run([A,initA])
    print "####train####"
    for i in range(step):
        # y_, _, A, initA, v_ac = sess.run([yy, train, trans_matrix,init_trans, v_accuracy],{is_train:1})
        # _, ret = viterbi(y_, A, initA)

        # ctrain.append(v_ac)
        # ax.append(i)
        # ay.append(sess.run(accuracy, {y_viterbi: ret.reshape(-1)}))
        print i
        _,tf_v, y_ = sess.run([train, viterbi_sequence, y])
        y_ = np.reshape(y_,[-1])
        # print tf_v
        # print y_
        batch_sequence_lengths = max_stc_size * train_batch_size
        mask = (np.expand_dims(np.arange(batch_sequence_lengths), axis=0) < np.expand_dims(batch_sequence_lengths,
                                                                                           axis=1))
        total_labels = np.sum(batch_sequence_lengths)
        correct_labels = np.sum((y_ == tf_v) * mask)
        ac = 1.0 * correct_labels / float(total_labels)
        # print ac
        ax.append(i)
        ay.append(ac)
    # sess.run(train_initializer)
    print "####test####"
    sess.run(test_initializer)
    ret = []
    for i in range(10):
        # bx.append(i)
        # y_ , A, initA= sess.run([viterbi_sequence, trans_matrix, init_trans])
        # ctest.append(sess.run(accuracy1))
        # _, ret = viterbi(y_, A, initA)
        # by.append(sess.run(accuracy, {y_viterbi: ret.reshape(-1)}))
        # print i
        tf_v,y_ = sess.run([viterbi_sequence,y])
        y_ = np.reshape(y_, [-1])
        # print y_
        batch_sequence_lengths = max_stc_size * train_batch_size
        mask = (np.expand_dims(np.arange(batch_sequence_lengths), axis=0) < np.expand_dims(batch_sequence_lengths,
                                                                                           axis=1))
        total_labels = np.sum(batch_sequence_lengths)
        correct_labels = np.sum((y_ == tf_v) * mask)
        ac = 1.0 * correct_labels / float(total_labels)
        # print ac
        bx.append(i)
        by.append(ac)
        ret.append(tf_v)


# plt.plot(ax, ay, 'r-',bx, by,'b-',ax, ctrain, 'g-', bx, ctest,'y-')
# plt.ylim(np.arange(0,1))
plt.plot(ax,ay,'r-',bx,by,'b-')
plt.savefig("result2.png")


