from __future__ import print_function
import math
import os
import random
import cPickle as pickle
from time import time

import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split


class BaseNN(object):
    """
    the base class for ANN, CNN and RNN, provide methods for train the model
    """
    def init(self):
        pass

    def get_parameters(self):
        pass

    def add_layer(self, inputs, in_size, out_size, n_layer, act=tf.nn.relu):
        layer_name = 'layer%s' % n_layer
        with tf.name_scope(layer_name):
            W = tf.Variable(tf.truncated_normal([in_size,out_size],
                stddev=1.0/math.sqrt(float(in_size))), name='Weights')
            b = tf.Variable(tf.zeros([out_size]),name='biases')
            out_put = act(tf.matmul(inputs,W) + b)
            out_dropout = tf.nn.dropout(out_put, self.dropout_keep_prob)
        return out_dropout

    def loss(self,y,y_):
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
        return cross_entropy
    
    def train(self, loss):
        self.global_step = tf.Variable(0, name='global_step')
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return train_op

    def fit(self, X, y, num_epochs=100, batch_size=512, max_step=10000,
            keep_drop=0.5, save_path=None, verbose=False):
        data_size = len(y)
        # Y = np.array([[1 if j==i else 0 for j in range(self.num_classes)] for i in y])
        X, Y = self.next_batch(X, y, data_size, 0, self.num_classes)
        for epoch in range(num_epochs):
            self.shuffle(X, y)
            step_size = data_size/batch_size
            for step in range(step_size):
                batch_corpus, batch_labels = self.next_batch(X, y, 
                    batch_size, step, self.num_classes)
                feed_dict = {self.input_x : batch_corpus, 
                                self.input_y : batch_labels, 
                                self.dropout_keep_prob: keep_drop}
                _, step = self.sess.run([self.train_op, self.global_step], feed_dict=feed_dict)
                if step > max_step:
                    return self
            acc, loss_ = self.sess.run([self.accuracy, self.loss], 
                        feed_dict={self.input_x: X, self.input_y: Y, self.dropout_keep_prob: 1.0})
            if verbose:
                print("epoch:%d  loss:%f  acc: %f" %(epoch, loss_, acc) )
            if self.stop_iter(acc, epoch):
                break
            
        if save_path:
            self.save_model(save_path)
        return self

    def predict(self, X_test):
        y_pre = self.sess.run(self.predictions, 
                    feed_dict={self.input_x: X_test, self.dropout_keep_prob: 1.0})
        return y_pre

    def predict_proba(self, X_test):
        proba = self.sess.run(self.output,
                    feed_dict={self.input_x: X_test, self.dropout_keep_prob: 1.0})
        return proba

    def shuffle(self, X_train, y_train):
        data = zip(X_train, y_train)
        random.shuffle(data)
        X, y = zip(*data)
        return X, y


    def stop_iter(self, acc, epoch):
        if epoch == 0:
            self.acc_list = []
        self.acc_list.append(acc)
        max_acc = max(self.acc_list)
        if len(self.acc_list) - self.acc_list.index(max_acc) > 3:
            return True
        return False

    def next_batch(self, X_train, y_train, batch_size, step, num_class, shuffle=True):
        total = len(y_train)

        if (step+1)*batch_size > (total -2) :
            start = (step+1)*batch_size % total
        else:
            start = step*batch_size
        end = start + batch_size
        try:
            corpus_batch = X_train[start:end]
            labels_batch = np.array([[1 if j==i else 0 for j in range(num_class)] for i in y_train[start:end]])
        except IndexError:
            corpus_batch = X_train[start:]
            labels_batch = np.array([[1 if j==i else 0 for j in range(num_class)] for i in y_train[start:]])

        return corpus_batch,labels_batch

