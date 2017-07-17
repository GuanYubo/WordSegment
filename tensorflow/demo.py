#code:utf-8
from __future__ import print_function
import sys
from optparse import OptionParser
from time import time
import codecs
from sklearn.cross_validation import train_test_split
from sklearn import metrics

import numpy as np

from NNLM import TextCNN
from NNLM import TextRNN
from NNLM import TextANN
from NNLM import CNN
from NNLM import ANN
from NNLM import RNN
from clean import Clean


def read_sample(binary=True):
    path = '../guanyubo/data/'
    corpus = np.array([line.strip() for line in open(path+'binary-words.txt', "r")])
    n = 1
    labels = []
    for line in codecs.open(path+'binary-all-labels.txt', 'r', 'utf-8-sig'):
        try:
            label = int(line.strip())
            if label not in [0, 1, 2, 3, 4, 5, 6, 7]:
                print(label)
                print(n)
            if label == 0:
                labels.append(label)
            elif binary:
                labels.append(1)
            else:
                labels.append(label)
            n += 1
        except Exception, e:
            print(e)
            print(n)
            sys.exit(1)
    labels = np.array(labels)
    return corpus, labels


def train_ann(corpus, labels):
    layer_structure = [3000, 50, 2]
    ann = TextANN(layer_structure, 0.0, vocabulary=None)
    tdm_out = ann.fit_corpus(corpus, labels)
    X_train, X_test, y_train, y_test = train_test_split(tdm_out, labels, test_size=0.1, random_state=1)
    del corpus, labels
    if opts.save:
        save_path = 'deploy/model_ann'
    else:
        save_path = None
    ann.fit(X_train, y_train, keep_drop=0.9, verbose=True, save_path=save_path)
    pre = ann.predict(X_test)
    print(metrics.classification_report(y_test,pre) )
    print('the precision is %f'%np.mean(pre == y_test))
    print(metrics.confusion_matrix(y_test,pre) )


def train_cnn(corpus, labels):
    cnn = TextCNN(
            sequence_length=20,
            num_classes=2,
            vocab_size=5000,
            vocabulary=None,
            embedding_size=128,
            filter_sizes=[1,2,3],
            num_filters=36,
            l2_reg_lambda=0.0)

    tdm_out = cnn.fit_corpus(corpus, labels)
    X_train, X_test, y_train, y_test = train_test_split(tdm_out, labels, test_size=0.1, random_state=1)
    del corpus, labels
    if opts.save:
        save_path = 'deploy/model_cnn'
    else:
        save_path = None
    t0 = time()
    cnn.fit(X_train, y_train, num_epochs=50, keep_drop=0.5, batch_size=512, max_step=5000, verbose=True, save_path=save_path)
    print('train the model, cost time:%f' % (time()-t0))
    t0 = time()
    pre = cnn.predict(X_test)
    t1 = time()
    print(metrics.classification_report(y_test, pre))
    print('the precision is %f  cost time: %f' % (np.mean(pre == y_test), (t1 - t0)))
    print(metrics.confusion_matrix(y_test, pre))


def train_rnn(corpus, labels):
    rnn = TextRNN(
            sequence_length=20,
            num_classes=2,
            embedding_size=128,
            vocab_size=5000,
            vocabulary=None,
            l2_reg_lambda=0.8,
            pool='max')

    tdm_out = rnn.fit_corpus(corpus, labels, select='tf')
    X_train, X_test, y_train, y_test = train_test_split(tdm_out, labels, test_size=0.1, random_state=32)
    del corpus, labels
    if opts.save:
        save_path = 'deploy/model_rnn'
    else:
        save_path = None
    rnn.fit(X_train, y_train, num_epochs=100, keep_drop=0.9, verbose=True, save_path=save_path)
    t0 = time()
    pre = rnn.predict(X_test)
    t1 = time()
    print(metrics.classification_report(y_test,pre) )
    print('the precision is %f cost time: %f'%(np.mean(pre==y_test), (t1-t0)) )
    print(metrics.confusion_matrix(y_test,pre) )

def pre_ann(raw_msg):
    path = 'deploy/model_ann'
    ann = ANN(path)
    text_clean = Clean()
    t0 = time()
    input_words = text_clean.clean(raw_msg)
    x = np.array([' '.join(input_words)])
    proba = ann.predict_proba(x)
    label = proba[0].tolist().index(max(proba[0]))
    print('the predict label is %d'%label)
    print(proba[0])
    print('cost time:%f\n'%(time()-t0))


def pre_cnn(raw_msg):
    path = 'deploy/model_cnn'
    cnn = CNN(path)
    text_clean = Clean()
    t0 = time()
    input_words = text_clean.clean(raw_msg)
    x = np.array([' '.join(input_words)])
    proba = cnn.predict_proba(x)
    label = proba[0].tolist().index(max(proba[0]))
    print('the predict label is %d'%label)
    print(proba[0])
    print('cost time:%f\n'% (time()-t0))


def pre_rnn(raw_msg):
    path = 'deploy/model_rnn'
    rnn = RNN(path)
    text_clean = Clean()
    t0 = time()
    input_words = text_clean.clean(raw_msg)
    x = np.array([' '.join(input_words)])
    proba = rnn.predict_proba(x)
    label = proba[0].tolist().index(max(proba[0]))
    print('the predict label is %d'%label)
    print(proba[0])
    print('cost time:%f\n' % (time()-t0))


def main():
    if opts.predict:
        try:
            raw_msg = sys.argv[2]
        except IndexError:
            print('please input the message for predict\n')
            sys.exit(1)
        if opts.ann:
            pre_ann(raw_msg)
        if opts.cnn:
            pre_cnn(raw_msg)
        if opts.rnn:
            pre_rnn(raw_msg)
    else:
        corpus, labels = read_sample(True)
        if opts.ann:
            train_ann(corpus, labels)
        if opts.cnn:
            train_cnn(corpus, labels)
        if opts.rnn:
            train_rnn(corpus, labels)

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')

    op = OptionParser()
    op.add_option("-c", "--cnn",
                  action="store_true", dest="cnn",
                  help="train model with cnn")
    op.add_option("-r", "--rnn",
                  action="store_true", dest="rnn",
                  help="train model with rnn")
    op.add_option("-a", "--ann",
                  action="store_true", dest="ann",
                  help="train model with ann")
    op.add_option("-p", "--predict",
                  action="store_true", dest="predict",
                  help="predict the message")
    op.add_option("-s", "--save",
                  action="store_true", dest="save",
                  help="save the model")
    (opts, args) = op.parse_args()

    if opts.cnn or opts.ann or opts.rnn:
        main()
    else:
        op.print_help()
        op.error('missing option')
        sys.exit(1)



