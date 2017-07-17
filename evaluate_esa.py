# coding: utf-8
from __future__ import division
import sys
import numpy as np
import codecs


# precision, recall, F1-score

# convert file into cut list
def convertW(infile):
    f = open(infile, 'r')
    text = f.read().replace('\n','')
    f.close()

    result = [0]
    for w in text.strip('\n').split('|'):
        result.append(len(w) + result[-1])
    return result[1:-1]

# evaluation
def evaluateW(seg_pre, seg_std):
    correct_num = 0
    error_num = 0
    std_p = 0
    last_same = 1

    for p in seg_pre:
        try:
            if p == seg_std[std_p]:
                if last_same == 1:
                    correct_num += 1
                else:
                    error_num += 1
                    last_same = 1
                std_p += 1
            elif p < seg_std[std_p]:
                error_num += 1
                last_same = 0
            elif p > seg_std[std_p]:
                error_num += 1
                while p > seg_std[std_p]:
                    std_p += 1
                if p == seg_std[std_p]:
                    std_p += 1
                    last_same = 1
                else:
                    last_same = 0
        except:
            print ('------')

    p = correct_num / (correct_num + error_num)
    r = correct_num / len(seg_std)
    f = 2 * p * r / (p + r)
    # print(len(seg_pre), len(seg_std), correct_num, error_num)
    return p, r, f

# transfer cut list into labeling list
def GetList(words):
    OutState = str()
    if len(words) == 1:
        OutState = 'S'

    elif len(words) == 2:
        OutState = 'BE'

    else:
        Snum = len(words) - 2
        Slist = 'M' * Snum
        OutState = 'B' + Slist + 'E'
    return OutState

# transfer file into labeling list
def convertT(infile, raw):
    f = open(infile, 'r')
    text = f.read().replace('\n','|').replace('||', '|')
    f.close()

    if raw:
        states = str()
        for w in text.strip('\n').split('|'):
            if w == '':
                continue
            w = w.decode('utf-8','ignore')
            states = states + GetList(w)
        return states
    else:
        return text

# evaluate labeling list
def evaluateT(prestr, stdstr):
    # print(len(prestr), type(prestr), len(stdstr), type(stdstr))
    correct_num = 0
    error_num = 0

    for i in range(len(prestr)):
        if prestr[i] == stdstr[i]:
            correct_num += 1
        else:
            error_num += 1
    p = correct_num / (correct_num + error_num)
    # print(correct_num, error_num)
    return p

# transfer labeling series into numeric series
def convert(seg):
    result = [0]
    for w in seg:
        result.append(len(w) + result[-1])
    return result[1:]

# main evaluation func
def evaluate(std, seg):
    seg_std = convert(std)
    seg_pre = convert(seg)
    assert seg_std[-1] == seg_pre[-1]
    l = len(seg_std)
    correct_num = 0
    error_num = 0
    std_p = 0
    last_same = 1
    for p in seg_pre:
        if p == seg_std[std_p]:
            if last_same == 1:
                correct_num += 1
            else:
                error_num += 1
                last_same = 1
            std_p += 1
        elif p < seg_std[std_p]:
            error_num += 1
            last_same = 0
        elif p > seg_std[std_p]:
            error_num += 1
            while p > seg_std[std_p]:
                std_p += 1
            if p == seg_std[std_p]:
                std_p += 1
                last_same = 1
            else:
                last_same = 0
    p = correct_num / (correct_num+error_num)
    r = correct_num / l
    f = 2*p*r / (p+r)
    return p, r, f

# evaluation for esa results
def evaluate_esa(std_file, pre_file):
    with codecs.open(std_file,'rb') as f:
        std_line = f.readlines()
    with codecs.open(pre_file,'rb') as f:
        pre_line = f.readlines()
    std_seg = []
    pre_seg = []
    for std,pre in zip(std_line,pre_line):
        s = std.strip().split('|')
        p = pre.strip().split('|')
        if len(''.join(s)) != len(''.join(p)):
            print ('-:\t%s\n  \t%s\n'%(std,pre))
            continue
        else:
            print ('\t%s\n\t%s\n'%(std,pre))
        std_seg.extend(s)
        pre_seg.extend(p)
    p,r,f = evaluate(std_seg,pre_seg)
    return (p,r,f)



if __name__ == '__main__':
    (precision,recall,Fscore) = evaluate_esa('data/std.txt', 'data/pre.txt')

