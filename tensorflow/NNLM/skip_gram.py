# code:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import random

import numpy as np
import tensorflow as tf
import cPickle as pickle


class SkipGram(object):
    def __init__(self, embedding_size, vocabulary_size):
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size

    def build_dataset(self, words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        self.dictionary = dict()
        for word, _ in count:
            self.dictionary[word] = len(self.dictionary)
        self.data = list()
        unk_count = 0
        for word in words:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            self.data.append(index)
        count[0][1] = unk_count
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def fit(self, words, batch_size=128, num_steps=100000, num_skips=2, skip_window=2):
        self.build_dataset(words)
        with tf.Graph().as_default():
            with tf.Session() as session:
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

                with tf.device('/cpu:0'):
                    embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                    # Construct the variables for the NCE loss
                    nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],stddev=1.0 / math.sqrt(self.embedding_size)))
                    nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

                # Compute the average NCE loss for the batch.
                # tf.nce_loss automatically draws a new sample of the negative labels each
                # time we evaluate the loss.
                num_sampled = 64    # Number of negative examples to sample.
                loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,num_sampled, self.vocabulary_size))
                optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

                # Compute the cosine similarity between minibatch examples and all embeddings.
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm
                init = tf.initialize_all_variables()
                init.run()
                print("Initialized")
                average_loss = 0
                self.data_index = 0
                for step in range(num_steps+1):
                    batch_inputs, batch_labels = self._generate_batch(batch_size, num_skips, skip_window)
                    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
                    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += loss_val

                    if step % 2000 == 0:
                        if step > 0:
                            average_loss /= 2000
                            print("Average loss at step ", step, ": ", average_loss)
                            average_loss = 0
                final_embeddings = normalized_embeddings.eval()
                pickle.dump(final_embeddings,open('embedding.pkl','wb'))
                dic_name = 'dic_%d'%self.vocabulary_size
                pickle.dump(self.dictionary,open(dic_name,'wb'))

    
    def _generate_batch(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        return batch, labels


