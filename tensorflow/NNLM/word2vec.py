from __future__ import print_function
import sys
import cPickle as pickle
import random

import numpy as np


class WordVec(object):
	def __init__(self, path):
		self.embedding = pickle.load(open(path+'/embedding.pkl','rb'))
		self.dictionary = pickle.load(open(path+'/dic.pkl','rb'))
		embedding_shape = np.shape(self.embedding)
		assert embedding_shape[0] == len(self.dictionary)

	def near(self, word ,n):
		try:
			valid_example = self.dictionary[word]
		except KeyError:
			print('the word %s is not in the vocabulary'%word)
			return
		reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
		valid_embeddings = self.embedding[valid_example]
		similarity = np.dot(valid_embeddings, np.transpose(self.embedding))
		top_k = n
		nearest = (-similarity[:]).argsort()[1:top_k+1]
		log_str = 'Nearest to %s:' % word
		for k in range(top_k):
			close_word = reverse_dictionary[nearest[k]]
			log_str = '%s %s ' % (log_str, close_word)
		print(log_str)
	
	def random(self, n_words, n_nearest):
		voca_list = range(len(self.dictionary))
		words_list = random.sample(voca_list, n_words)
		reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
		for index in words_list:
			word = reverse_dictionary[index]
			self.near(word, n_nearest)

	def similary(self, word1, word2):
		try:
			w1 = self.embedding[self.dictionary[word1]]
		except KeyError:
			print('the word %s is not in the vocabulary'%word1)
			return
		try:
			w2 = self.embedding[self.dictionary[word2]]
		except KeyError:
			print('the word %s is not in the vocabulary'%word2)
			return
		similarity = self.cosdis(w1, w2)
		print('\nthe cossimilarity of the %s and %s is %f\n' %(word1, word2, similarity))
	
	def cosdis(self, w1, w2):
		w1 = np.mat(w1)
		w2 = np.mat(w2)
		num = float(w1*w2.T)
		denom = np.linalg.norm(w1)*np.linalg.norm(w2)
		return 0.5+0.5*(num/denom)

