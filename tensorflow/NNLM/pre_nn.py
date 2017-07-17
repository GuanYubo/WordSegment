from __future__ import print_function
import cPickle as pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from text_nn import TextANN
from text_nn import TextCNN
from text_nn import TextRNN
from wordembedding import EmbeddingVectorizer


class ANN(TextANN):
	def __init__(self, path):
		self.path = path
		parameters = pickle.load(open(path+'/parameters.pkl','rb'))
		self.layer_structure = parameters['layer_structure']
		self.input_size = self.layer_structure[0]
		self.num_classes = self.layer_structure[-1]
		self.l2_lambda = parameters['l2_lambda']
		self.vocabulary = pickle.load(open(path+'/dic.pkl','rb'))
		self.count = CountVectorizer(vocabulary=self.vocabulary)
		with tf.Graph().as_default():
			super(ANN, self).init()
			self.saver = tf.train.Saver()
			self.sess = tf.Session()
			self.saver.restore(self.sess, self.path+'/model_ann.ckpt')

	def predict(self,input_words):
		words = self.count.fit_transform(input_words).todense()
		pre = self.sess.run(self.predictions, feed_dict={self.input_x: words, self.dropout_keep_prob: 1.0})
		return pre

	def predict_proba(self,input_words):
		words = self.count.fit_transform(input_words).todense()
		proba = self.sess.run(self.output,feed_dict={self.input_x: words, self.dropout_keep_prob: 1.0})
		return proba


class CNN(TextCNN):
	"""
	CNN extends TextCNN for predict message
	"""
	def __init__(self, path):
		parameters = pickle.load(open(path+'/parameters.pkl','rb'))
		self.vocabulary = pickle.load(open(path+'/dic.pkl','rb'))
		self.sequence_length = parameters['sequence_length']
		self.num_classes = parameters['num_classes']
		self.vocab_size = len(self.vocabulary)
		self.embedding_size = parameters['embedding_size']
		self.filter_sizes = parameters['filter_sizes']
		self.num_filters = parameters['num_filters']
		self.l2_reg_lambda = parameters['l2_reg_lambda']
		with tf.Graph().as_default():
			super(CNN, self).init()
			self.saver = tf.train.Saver()
			self.sess = tf.Session()
			self.saver.restore(self.sess, path+'/model_cnn.ckpt')

	def predict(self, input_words):
		words = self.fit_corpus(input_words)
		pre = self.sess.run(self.predictions, feed_dict={self.input_x: words, self.dropout_keep_prob: 1.0})
		return pre
	
	def predict_proba(self, input_words):
		words = self.fit_corpus(input_words)
		proba = self.sess.run(self.output, feed_dict={self.input_x: words, self.dropout_keep_prob: 1.0})
		return proba


class RNN(TextRNN):
	"""
	CNN extends TextRNN for predict message
	"""
	def __init__(self, path):
		parameters = pickle.load(open(path+'/parameters.pkl','rb'))
		self.vocabulary = pickle.load(open(path+'/dic.pkl','rb'))
		self.vocab_size = len(self.vocabulary)
		self.sequence_length = parameters['sequence_length']
		self.num_classes = parameters['num_classes']
		self.embedding_size = parameters['embedding_size']
		self.l2_reg_lambda = parameters['l2_reg_lambda']
		self.pool = parameters['pool']
		with tf.Graph().as_default():
			super(RNN, self).init()
			self.saver = tf.train.Saver()
			self.sess = tf.Session()
			self.saver.restore(self.sess, path+'/model_rnn.ckpt')

	def predict(self, input_words):
		words = self.fit_corpus(input_words)
		pre = super(RNN, self).predict(words)
		return pre
	
	def predict_proba(self, input_words):
		words = self.fit_corpus(input_words)
		proba = super(RNN, self).predict_proba(words)
		return proba
