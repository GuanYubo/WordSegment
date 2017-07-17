from __future__ import print_function
import os
import math
import cPickle as pickle
from time import time

import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import train_test_split

from base_nn import BaseNN
from base_vectorizer import Vectorizer


class TextCNN(BaseNN, Vectorizer):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size=10000, vocabulary=None,
                 embedding_size=128, filter_sizes=[1, 2, 3], num_filters=24, l2_reg_lambda=0.0):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        Vectorizer.__init__(self, vocabulary, vocab_size)
        with tf.Graph().as_default():
            self.init()
            self.saver = tf.train.Saver()
            summary = tf.merge_all_summaries()
            self.sess = tf.Session()
            summary_writer = tf.train.SummaryWriter('logs/',self.sess.graph)
            init = tf.initialize_all_variables()
            self.sess.run(init)

    def get_parameters(self):
        parameters = {'sequence_length': self.sequence_length, 'num_classes': self.num_classes, 
                'vocab_size': self.vocab_size, 'embedding_size': self.embedding_size, 
                'filter_sizes': self.filter_sizes, 'num_filters': self.num_filters, 
                'l2_reg_lambda': self.l2_reg_lambda}
        return parameters

    def init(self):
        # Placeholders for input, output and dropout
        with tf.name_scope('input'):
            self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars_static = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded_static = tf.expand_dims(self.embedded_chars_static, -1)

        #with tf.device('/cpu:0'), tf.name_scope("embedding-nonstatic"):
        #   W = tf.Variable(self.embedding, name='W')
        #   self.embedded_chars_nonstatic = tf.nn.embedding_lookup(W, self.input_x)
        #   self.embedded_chars_expanded_nonstatic = tf.expand_dims(self.embedded_chars_nonstatic, -1)
        #self.embedded_chars_expanded = tf.concat(3, [self.embedded_chars_expanded_static, self.embedded_chars_expanded_nonstatic])

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope('conv'):
            pooled_outputs_static = []
            for i, filter_size in enumerate(self.filter_sizes):
            #with tf.name_scope("conv-maxpool-static-%s"%filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded_static, W, 
                        strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], 
                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs_static.append(pooled)
                
                # Combine all the pooled features
                num_filters_total = self.num_filters * len(self.filter_sizes)
                self.h_pool_static = tf.concat(3, pooled_outputs_static)
                self.h_pool_flat_static = tf.reshape(self.h_pool_static, [-1, num_filters_total])
        

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat_static, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, self.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.output = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.output, 1, name="predictions")
        
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            #losses = -tf.reduce_sum(self.input_y*tf.log(tf.clip_by_value(self.output,1e-10,1.0)))
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss
        # Train step
        with tf.name_scope('train'):
            self.train_op = self.train(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    


    def save_model(self, save_path):
        try:
            mkdir =  save_path
            os.makedirs(mkdir)
        except OSError:
            mkdir = save_path + str(time())
            os.makedirs(mkdir)
            print('the path %s has been exist, save the model to %s'
                    %(save_path, mkdir ))
        self.path = self.saver.save(self.sess, mkdir+'/model_cnn.ckpt')
        parameters = self.get_parameters()
        embedding = self.sess.run('embedding-static/W:0')
        pickle.dump(embedding, open(mkdir+'/embedding.pkl','wb'))
        pickle.dump(parameters, open(mkdir+'/parameters.pkl','wb'))
        pickle.dump(self.vocabulary, open(mkdir+'/dic.pkl','wb'))



class TextANN(BaseNN):
    """
    A ANN for text classification
    with a input layer,
    several hidden layers,use the relu active function,
    a output layer using softmax function.
    """
    def __init__(self, layer_structure, l2_lambda=0.0, vocabulary=None):
        self.layer_structure = layer_structure
        self.input_size = layer_structure[0]
        self.num_classes = layer_structure[-1]
        self.l2_lambda = l2_lambda
        self.vocabulary = vocabulary
        with tf.Graph().as_default():
            self.init()
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            init = tf.initialize_all_variables()
            self.sess.run(init)

    def init(self):
        with tf.name_scope('inputs'):
            self.input_x = tf.placeholder(tf.float32,shape=(None, self.input_size), name='corpus')
            self.input_y = tf.placeholder(tf.float32,shape=(None, self.num_classes), name='labels')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.l2_loss = tf.constant(0.0)
        self.output = self.inference(self.input_x, self.layer_structure)
        self.predictions = tf.argmax(self.output, 1, name='predictions')
        with tf.name_scope('loss'):
            losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.input_y))
            self.loss = losses + self.l2_lambda * self.l2_loss
        self.train_op = self.train(self.loss)
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        

    def add_layer(self,inputs,in_size,out_size,n_layer,act=tf.nn.relu):
        layer_name = 'layer%s' % n_layer
        with tf.name_scope(layer_name):
            W = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=1.0/math.sqrt(float(in_size))),name='Weights')
            b = tf.Variable(tf.zeros([out_size]),name='biases')
            out_put = act(tf.matmul(inputs,W) + b)
            out_dropout = tf.nn.dropout(out_put, self.dropout_keep_prob)
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
        return out_dropout

    def inference(self,inputs,layer_structure):
        layer_num = len(layer_structure)
        assert layer_num > 1
        hidden = inputs 
        for layer in range(layer_num-2):
            hidden = self.add_layer(hidden, layer_structure[layer], layer_structure[layer+1], layer+1)

        layer_output = self.add_layer(hidden, layer_structure[layer_num-2], self.num_classes, layer_num-1, tf.nn.softmax)
        return layer_output



    def get_parameters(self):
        parameters = {'layer_structure': self.layer_structure, 'l2_lambda': self.l2_lambda}
        return parameters

    def get_vocabulary(self, corpus, labels):
        vect = CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1,1), binary=True, vocabulary=self.vocabulary).fit(corpus)
        dic_self = vect.get_feature_names()                                 
        tdm = vect.fit_transform(corpus)
        sel_f = SelectKBest(chi2, self.input_size)
        tdm_out = sel_f.fit_transform(tdm, labels)
        index = sel_f.fit(tdm, labels).get_support(indices=True)
        dic_select = {}
        for i in index:
            dic_select[dic_self[i]] = len(dic_select)
        return dic_select

    def fit_corpus(self, corpus, labels=None):
        if not self.vocabulary or self.input_size != len(self.vocabulary):
            self.vocabulary = self.get_vocabulary(corpus, labels)
        count = CountVectorizer(min_df=1, max_df=0.5, binary=True, vocabulary=self.vocabulary)
        tdm = count.fit_transform(corpus)
        return tdm

    def save_model(self, save_path):
        try:
            mkdir =  save_path
            os.makedirs(mkdir)
        except OSError:
            mkdir = save_path + str(time())
            os.makedirs(mkdir)
            print('the path %s has been exist, save the model to %s'
                    %(save_path, mkdir ))
        self.path = self.saver.save(self.sess, mkdir+'/model_ann.ckpt')
        parameters = self.get_parameters()
        pickle.dump(parameters, open(mkdir+'/parameters.pkl','wb'))
        pickle.dump(self.vocabulary, open(mkdir+'/dic.pkl','wb'))

    def next_batch(self, X_train, y_train, batch_size, step, num_class):
        total = len(y_train)

        if (step+1)*batch_size > (total -2) :
            start = (step+1)*batch_size % total
        else:
            start = step*batch_size
        end = start + batch_size
        try:
            corpus_batch = X_train[start:end].todense()
            labels_batch = np.array([[1 if j==i else 0 for j in range(num_class)] for i in y_train[start:end]])
        except IndexError:
            corpus_batch = X_train[start:].todense()
            labels_batch = np.array([[1 if j==i else 0 for j in range(num_class)] for i in y_train[start:]])

        return corpus_batch,labels_batch
    
    def predict(self, X_test):
        y_pre = self.sess.run(self.predictions, 
                    feed_dict={self.input_x: X_test.todense(), self.dropout_keep_prob: 1.0})
        return y_pre

    def predict_proba(self, X_test):
        proba = self.sess.run(self.output,
                    feed_dict={self.input_x: X_test.todense(), self.dropout_keep_prob: 1.0})
        return proba



class TextRNN(BaseNN, Vectorizer):
    """
    """
    def __init__(self, sequence_length, num_classes, 
            embedding_size=128, vocab_size=10000, vocabulary=None, l2_reg_lambda=0.0, pool='last'):
        Vectorizer.__init__(self, vocabulary, vocab_size)
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.l2_reg_lambda=l2_reg_lambda
        self.pool = pool
        with tf.Graph().as_default():
            self.init()
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            init = tf.initialize_all_variables()
            self.sess.run(init)

    def init(self):
        with tf.name_scope('input'):
            self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)

        with tf.name_scope('lstm-pool'):
            output_h = []
            self.out = tf.placeholder(tf.float32, [None, self.sequence_length])
            words = self.embedded_chars[:,0,:]
            h, state = self.lstm_cell(words, self.out, self.out)
            for i in range(1,self.sequence_length):
                words = self.embedded_chars[:,i,:]
                h, state = self.lstm_cell(words, h, state)
                output_h.append(h)
            # last-pooling
            if self.pool == 'last':
                self.h_pool = output_h[-1]
            # mean-pooling
            elif self.pool == 'mean':
                self.h_pool = tf.reduce_mean(output_h, 0)
            # max-pooling
            elif self.pool == 'max':
                self.h_pool = tf.reduce_max(output_h, 0)

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool, self.dropout_keep_prob)

        with tf.name_scope('output'):
            W = tf.get_variable("W", shape=[self.sequence_length, self.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.output = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.output, 1, name="predictions")

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda*l2_loss

        with tf.name_scope('train'):
            self.train_op = self.train(self.loss)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name = 'accuracy')

    
    def lstm_cell(self, input_words, h, state):
        #input gate
        Wi = tf.Variable(tf.truncated_normal([self.embedding_size, self.sequence_length], -0.1, 0.1))
        Ui = tf.Variable(tf.truncated_normal([self.sequence_length, self.sequence_length], -0.1, 0.1))
        bi = tf.Variable(tf.zeros([1, self.sequence_length]))
        #forget gate
        Wf = tf.Variable(tf.truncated_normal([self.embedding_size, self.sequence_length], -0.1, 0.1))
        Uf = tf.Variable(tf.truncated_normal([self.sequence_length, self.sequence_length], -0.1, 0.1))
        bf = tf.Variable(tf.zeros([1, self.sequence_length]))
        #memory cell
        Wc = tf.Variable(tf.truncated_normal([self.embedding_size, self.sequence_length], -0.1, 0.1))
        Uc = tf.Variable(tf.truncated_normal([self.sequence_length, self.sequence_length], -0.1, 0.1))
        bc = tf.Variable(tf.zeros([1, self.sequence_length]))
        #output gate
        Wo = tf.Variable(tf.truncated_normal([self.embedding_size, self.sequence_length], -0.1, 0.1))
        Uo = tf.Variable(tf.truncated_normal([self.sequence_length, self.sequence_length], -0.1, 0.1))
        bo = tf.Variable(tf.zeros([1, self.sequence_length]))
        #lstm cell computation
        input_gate = tf.sigmoid(tf.matmul(input_words, Wi) + tf.matmul(h, Ui) + bi)
        forget_gate = tf.sigmoid(tf.matmul(input_words, Wf) + tf.matmul(h, Uf) + bf)
        update = tf.tanh(tf.matmul(input_words, Wc) + tf.matmul(h, Uc) + bc)
        state = forget_gate*state + input_gate*update
        output_gate = tf.sigmoid(tf.matmul(input_words, Wo) + tf.matmul(h, Uo) + bo)
        output_h = output_gate*tf.tanh(state)
        return output_h, state

    def fit(self, X, y, num_epochs=100, batch_size=512,
            keep_drop=0.5, save_path=None, verbose=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        data_size = len(y_train)
        for epoch in range(num_epochs):
            self.shuffle(X_train, y_train)
            step_size = data_size/batch_size
            for step in range(step_size):
                batch_corpus, batch_labels = self.next_batch(X_train, y_train, 
                    batch_size, step, self.num_classes)
                feed_dict = {self.input_x : batch_corpus, 
                                self.input_y : batch_labels, 
                                self.dropout_keep_prob: keep_drop,
                                self.out: np.zeros_like(batch_corpus, dtype=float)}
                self.sess.run(self.train_op, feed_dict=feed_dict)
            X_, y_ = self.next_batch(X_train, y_train, data_size, 1, self.num_classes)
            acc, loss_ = self.sess.run([self.accuracy, self.loss], 
                        feed_dict={self.input_x: X_, self.input_y: y_, 
                            self.dropout_keep_prob: 1.0, self.out: np.zeros_like(X_, dtype=float)})
            if verbose:
                print("epoch:%d  loss:%f  acc: %f" %(epoch, loss_, acc ), end='  ')

            X_, y_ = self.next_batch(X_test, y_test, len(y_test), 1, self.num_classes)
            pre = self.sess.run(self.predictions, 
                    feed_dict={self.input_x: X_test, self.dropout_keep_prob: 1.0, self.out:np.zeros_like(X_test, dtype=float)})
            precision = np.mean(pre==y_test)
            print('precision:%f\n'%precision)
            if self.stop_iter(precision, epoch):
                break
        if save_path:
            self.save_model(save_path)
        return self

    def predict(self, X_test):
        y_pre = self.sess.run(self.predictions, 
                    feed_dict={self.input_x: X_test, self.dropout_keep_prob: 1.0,
                        self.out: np.zeros_like(X_test, dtype=float)})
        return y_pre

    def predict_proba(self, X_test):
        proba = self.sess.run(self.output,
                    feed_dict={self.input_x: X_test, self.dropout_keep_prob: 1.0,
                        self.out: np.zeros_like(X_test, dtype=float)})
        return proba

    def save_model(self, save_path):
        try:
            mkdir =  save_path
            os.makedirs(mkdir)
        except OSError:
            mkdir = save_path + str(time())
            os.makedirs(mkdir)
            print('the path %s has been exist, save the model to %s'
                    %(save_path, mkdir ))
        self.path = self.saver.save(self.sess, mkdir+'/model_rnn.ckpt')
        parameters = self.get_parameters()
        pickle.dump(parameters, open(mkdir+'/parameters.pkl','wb'))
        pickle.dump(self.vocabulary, open(mkdir+'/dic.pkl','wb'))

    def get_parameters(self):
        parameters = {'sequence_length': self.sequence_length, 'embedding_size': self.embedding_size,
                'num_classes': self.num_classes, 'l2_reg_lambda': self.l2_reg_lambda,
                'pool': self.pool}
        return parameters
