# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import sys
import os
import re
import logging
import time
import codecs
import numpy as np
import pandas as pd
import collections
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle
from evaluate_esa import evaluate_esa


# ESA
class ESA(object):

    def __init__(self, infile, stdfile, Lmax=4, LRV_index=1):
        self.path = infile[:-4]
        self.stdfile = stdfile
        self.corpus = self.ReadFile(infile)
        self.Lmax = Lmax
        self.LRV_index = LRV_index
        self.fv = collections.defaultdict(float)
        self.fs = collections.defaultdict(str)

        self.freq = self.Frequency()
        #self.entr = self.Entropy()
        #self.mean = self.Mean()

        #self.result = self.Select()
#               self.result = self.Dag()
        #self.output()
        #self.evaluate()

    def ReadFile(self, infile):
        clean_dot = re.compile(r"[”“，：:！!《》？?…：。·.,、;；’‘\[\]【】+-=_——()（）~·@￥$^%……&*#']".decode('utf-8'))
        clean_n = re.compile(r'\r\n'.decode('utf-8'))
        with codecs.open(infile,'rb') as f:
            text = f.readlines()
            text = [line.strip('\r\n') for line in text]
            text_all = '\n'.join(text).decode('utf-8')
            text_clean = clean_n.sub('\n', text_all)
            corpus = clean_dot.sub('\n', text_clean).split()
        return corpus

    def Frequency(self):
        # json: {key:w, value:{'left':{'xx':11,'xx':22,..},'right':{'yy':33,'yy':44,..}}
        ldic = {}
        rdic = {}
        dic = collections.defaultdict(int)
        for c in self.corpus:
            if c == '':
                continue
            for i in range(len(c)):
                j = i+1
                while j <= len(c): #j-i <= self.Lmax and
                    lw = ('B' if i == 0 else c[i-1])
                    rw = ('E' if j == len(c) else c[j])
                    w = c[i:j]
                    dic[w] += 1
                    try:
                        ldic[w][lw] += 1
                    except:
                        ldic[w] = collections.defaultdict(int)
                        ldic[w][lw] += 1
                    try:
                        rdic[w][rw] += 1
                    except:
                        rdic[w] = collections.defaultdict(int)
                        rdic[w][rw] += 1
                    j += 1
        F = {}
        for k in dic.keys():
            F[k] = {'freq':dic[k]+0.1,'left':ldic[k],'right':rdic[k]}
        pickle.dump(F,open(self.path+'_F.pkl','wb'))
        return F

    def Entropy(self):
        E = self.Frequency()
        for k in E.keys():
            E[k]['leftE'] = np.sum( [-(v/E[k]['freq'])*np.log2(v/E[k]['freq']) for v in E[k]['left'].values()] )
            E[k]['rightE'] = np.sum( [-(v/E[k]['freq'])*np.log2(v/E[k]['freq']) for v in E[k]['right'].values()] )
            del E[k]['left']
            del E[k]['right']
#                pickle.dump(E,open(self.path+'_E.pkl','wb'))
        return E

    def Mean(self):
        E = self.Entropy()
        M = collections.defaultdict(float)
        cnt = collections.defaultdict(int)
        for k in E.keys():
            cnt[len(k)] += 1
            M[(len(k), 'freq')] += E[k]['freq']
            M[(len(k), 'left')] += E[k]['leftE']
            M[(len(k), 'right')] += E[k]['rightE']
        for k in M.keys():
            M[k] /= cnt[k[0]]
#                pickle.dump(M,open(self.path+'_M.pkl','wb'))
        return M

    def IV(self, w):
        return (self.entr[w]['freq']/self.mean[(len(w), 'freq')])**len(w)

    def LRV(self, Sl, Sr):
        return (self.entr[Sl]['rightE']*self.entr[Sr]['leftE']/
                self.mean[(len(Sl)+1, 'right')]*self.mean[(len(Sr)+1, 'left')]) ** self.LRV_index  #指数是待估参数，衡量了LRV在CV中的权重

    def CV(self, Sl, Sr=''):
        if Sr == '':
            return self.IV(Sl)
        return self.IV(Sl) * self.IV(Sr) * self.LRV(Sl, Sr)

    # cut type one
    def Dag(self):
        result = []
        for c in self.corpus:
            if c == '':
                continue
            path = [0]
            j = len(c)
            while j > 0:
                value,word,k = max([(self.IV(c[j-i-1:j]), c[j-i-1:j], j-i-1) for i in range(self.Lmax) if j-i-1>=0], key=lambda x:x[0])
                j = k
                path.insert(1,word)
            result.append(path[1:])
        return result

        # cut type two
    def Segment(self,s):
        if self.fv[s] == 0:
            if len(s) == 1:
                self.fv[s] = self.IV(s)
                self.fs[s] = s
            else:
                Max = self.IV(s)
                Seg = s
                for i in range(1,len(s)):
                    self.Segment(s[0:i])
                    self.Segment(s[i:len(s)])
                    cv = self.fv[s[0:i]] * self.fv[s[i:len(s)]] * self.LRV(s[0:i], s[i:len(s)])
                    if Max < cv:
                        Max = cv
                        Seg = '|'.join([self.fs[s[0:i]], self.fs[s[i:len(s)]]])
                self.fv[s] = Max
                self.fs[s] = Seg
        return self

        # cut type two
    def Select(self):
        for c in self.corpus:
            if c.strip() != '':
                self.Segment(c.strip())
        result = [self.fs[c.strip()] for c in self.corpus if c.strip() != '']
        return result

    def output(self):
            timestamp = str(time.strftime('_%m_%d_%H_%M', time.localtime()))
            self.prefile = self.path+timestamp+'.txt'
            with codecs.open(self.prefile,'wb') as f:
                for r in self.result:
                    f.write('%s\n'%r.encode('utf-8'))

    def evaluate(self):
            (p,r,f) = evaluate_esa(self.stdfile,self.prefile)
            print ('\nPrecision:%0.6f, Recall:%0.6f, Fscore:%0.6f.\n'%(p,r,f))


if __name__ == '__main__':
    t0 = time.time()

    root = ''
    segfile = 'cityu_test_gold_raw.txt'
    stdfile = 'cityu_test_gold_std.txt'

    ewk = ESA(root+segfile, root+stdfile)

    print ('len(corpus)' + str(len(ewk.corpus)))
    print ('len(freq)' + str(len(ewk.freq)))
    print ('len(entr)' + str(len(ewk.entr)))
    print ('len(mean)' + str(len(ewk.mean)))

    print ('It costs: {}s .\n'.format(time.time() -t0))



