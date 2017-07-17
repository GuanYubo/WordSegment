import tensorflow as tf
import numpy as np
import tensorflow as tf


class AutoEncoder(object):
	def __init__(self, hidden_units):
		self.hidden_units = hidden_units
		with tf.Graph().as_default():
			self.creat_graph()

	def train(self, loss):
		self.global_step = tf.Variable(0)
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		grads_and_vars = optimizer.compute_gradients(loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
		return train_op

	def creat_graph(self):
		with tf.name_scope('input'):
			self.input_x = tf.placeholder(tf.float32, [None, self.sequence_length], name='input_x')
		with tf.name_scope('encode'):
			hidden = tf.add_layer(input_x, self.sequence_length, self.hidden_units)
		with tf.name_scope('decode'):
			self.output = tf.add_layer(hidden, self.hidden_units, self.sequence_length)
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(self.output, self.input_x)
			self.loss = tf.reduce_mean(losses)

	def add_layer(self, inputs,in_size, out_size):
		W = tf.Variable(tf.truncated_normal([in_size, out_size], 
			stddev=1.0/math.sqrt(float(in_size))), name='Weights')
		b = tf.Variable(tf.zeros([out_size]), name='biases')
		out_put = tf.nn.relu(tf.matmul(inputs, W) + b)
		return out_put

	def fit(self, X, batch_size=512, max_step=1000):
		step_size = data_size/batch_size
		while True:
			for step in range(step_size):
				batch = self.next_batch(X, batch_size, step)
				self.sess.run(self.train_op, feed_dict={self.input_x: batch})
				if step % 200 == 0:
					loss = self.sess.run(self.loss, feed_dict={self.input_x: batch})
				if step > max_step:
					self.save_model()
					return self
	
	def save_model(self):
		pass


	def next_batch(self, X_train, batch_size, step):
		total = len(y_train)

		if (step+1)*batch_size > (total -2) :
			start = (step+1)*batch_size % total
		else:
			start = step*batch_size
		end = start + batch_size
		try:
			corpus_batch = X_train[start:end]
		except IndexError:
			corpus_batch = X_train[start:]

		return corpus_batch


