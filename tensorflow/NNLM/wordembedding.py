import numpy as np

class EmbeddingVectorizer(object):
	def __init__(self, max_sentence_length, vocabulary):
		self.max_sentence_length = max_sentence_length
		self.vocabulary = self.check_vocabulary(vocabulary)

	def check_vocabulary(self, vocabulary):
		if isinstance(vocabulary, dict):
			return vocabulary
		else:
			vocab = {}
			for word in vocabulary:
				vocab[word] = len(vocab)
			return vocab

	def fit_transform(self, raw_document):
		word_vec_long = []
		for sentence in raw_document:
			word_vec = [self.vocabulary[w] if w in self.vocabulary else 0 for w in sentence.split(' ')]
			word_vec_long.append([word_vec[i] if i<len(word_vec) else 0 for i in range(self.max_sentence_length)])
		return np.array(word_vec_long)

	def transform(self,tdm):
		for word_vec in tdm:
			word_vec_long = [word_vec[i] if i<len(word_vec) else 0 for i in range(self.max_sentence_length)]
			yield word_vec_long
