import collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2


class Vectorizer(object):
	def __init__(self, vocabulary, vocab_size):
		self.vocabulary = self.check_vocabulary(vocabulary)
		self.vocab_size = vocab_size

	def check_vocabulary(self, vocabulary):
		if vocabulary is None:
			return None
		elif isinstance(vocabulary, dict):
			return vocabulary
		else:
			vocab = {}
			for w in vocabulary:
				vocab[word] = len(vocab)
			return vocab

	def get_vocabulary(self, corpus, labels):
		vect = CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1,1), binary=True, vocabulary=self.vocabulary).fit(corpus)
		dic_self = vect.get_feature_names()
		tdm = vect.fit_transform(corpus)
		sel_f = SelectKBest(chi2, self.vocab_size)
		tdm_out = sel_f.fit_transform(tdm, labels)
		index = sel_f.fit(tdm, labels).get_support(indices=True)
		dic_select = {}
		for i in index:
			dic_select[dic_self[i].encode()] = len(dic_select)
		return dic_select

	def build_vocabulary(self, words):
		count = []
		count.extend(collections.Counter(words).most_common(self.vocab_size))
		vocabulary = {}
		for w, _ in count:
			if w not in vocabulary:
				vocabulary[w] = len(vocabulary)
		return vocabulary

	def build_words(self, corpus):
		words = []
		for line in corpus:
			for w in line.split(' '):
				words.append(w)
		return words

	def fit_corpus(self, corpus, labels=None, select='tf'):
		if not self.vocabulary:
			if select == 'tf':
				words = self.build_words(corpus)
				self.vocabulary = self.build_vocabulary(words)
			elif select == 'chi2':
				self.vocabulary = self.get_vocabulary(corpus, labels)
			else:
				raise ValueError("the parameter select must be 'tf' or 'chi2'")
		word_vec_long = []
		for sentence in corpus:
			word_vec = [self.vocabulary[w] if w in self.vocabulary else 0 for w in sentence.split(' ')]
			word_vec_long.append([word_vec[i] if i < len(word_vec) else 0 for i in range(self.sequence_length)])
		return word_vec_long

