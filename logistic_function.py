#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: h12345jack
@file: logistic_function.py
@time: 2018/12/16
"""

import os
import sys
import re
import time
import json
import pickle
import logging
import math
import random
import argparse
import subprocess
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections import defaultdict

import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from sklearn import linear_model
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from common import DATASET_NUM_DIC
from fea_extra import FeaExtra

EMBEDDING_SIZE = 20



###obtain labels of train and test sets
###labels of positive sign and negtive sign  



def read_train_test_data(dataset, k):
    train_X = []
    train_y = []
    with open('./experiment-data/{}-train-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1)/2)  ###将标签1和-1转成了二分类1和0
            train_X.append((i, j))
            train_y.append(flag)
    test_X = []
    test_y = []
    with open('./experiment-data/{}-test-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1)/2)
            test_X.append((i, j))
            test_y.append(flag)
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
"""

###labels of positive edges of source->target  and negtive edges (将正样本取反，采样每条变对应的负样本)

def read_train_test_data(dataset, k):
    train_X = []
    train_y = []
    with open('./experiment-data/{}-train-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, _ = line.split()
            i = int(i)
            j = int(j)
            flag_p = 1
            flag_n = 0
            train_X.append((i, j))
            train_y.append(flag_p)     
            train_X.append((j, i))
            train_y.append(flag_n)   
            

    test_X = []
    test_y = []
    with open('./experiment-data/{}-test-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag_p = 1
            flag_n = 0
            test_X.append((i, j))
            test_y.append(flag_p)     
            test_X.append((j, i))
            test_y.append(flag_n) 
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
"""
###labels of positive edges of source->target  and negtive edges of source<-target (将样本分为正样本和负样本)
###依据：i->j,如果i>j:负样本    i<j:正样本
"""
def read_train_test_data(dataset, k):
    train_X = []
    train_y = []
    with open('./experiment-data/{}-train-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, _ = line.split()
            i = int(i)
            j = int(j)
            flag_p = 1
            flag_n = 0
            train_X.append((i, j))
            if i>j:
                train_y.append(flag_n)
            else:
                train_y.append(flag_p)   
            

    test_X = []
    test_y = []
    with open('./experiment-data/{}-test-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag_p = 1
            flag_n = 0
            test_X.append((i, j))
            if i>j:
                test_y.append(flag_n)
            else:
                test_y.append(flag_p)   
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)    

"""







def common_logistic(dataset, k, embeddings):
    train_X, train_y, test_X, test_y  = read_train_test_data(dataset, k)

    train_X1 = []
    test_X1 = []

    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    ###==logistic==============================
    logistic_function = linear_model.LogisticRegression()

    logistic_function.fit(train_X1, train_y)
    pred = logistic_function.predict(test_X1)
    pred_p = logistic_function.predict_proba(test_X1)



    pos_ratio =  np.sum(test_y) / test_y.shape[0]
    accuracy =  metrics.accuracy_score(test_y, pred)
    f1_score0 =  metrics.f1_score(test_y, pred)
    f1_score1 =  metrics.f1_score(test_y, pred, average='macro')
    f1_score2 =  metrics.f1_score(test_y, pred, average='micro')

    auc_score =  metrics.roc_auc_score(test_y, pred_p[:, 1])
    print("pos_ratio:", pos_ratio)
    print('accuracy:', accuracy)
    print("f1_score:", f1_score0)
    print("macro f1_score:", f1_score1)
    print("micro f1_score:", f1_score2)
    print("auc score:", auc_score)
###================================
    precision,recall,pr_thresholds = precision_recall_curve(test_y,pred_p[:,1])
    auprc_score = auc(recall,precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(test_y, pred_p[:,1])
    auc_score = auc(fpr, tpr)
    predicted_score = np.zeros(shape=(len(test_y), 1))
    predicted_score[pred_p[:,1] > threshold] = 1
    confusion_matri = confusion_matrix(y_true=test_y, y_pred=predicted_score)
    # print("confusion_matrix:", confusion_matri)
    f1 = f1_score(test_y, predicted_score)
    accuracy = accuracy_score(test_y,predicted_score)
    precision = precision_score(test_y,predicted_score)
    recall = recall_score(test_y,predicted_score)
    print("new auc_score:", auc_score)
    print('new accuracy:', accuracy)
    print("new precision:", precision)
    print("new macro f1_score:", f1_score1)
    print("new recall:", recall)
    print("new f1:", f1)
    print("new auprc_score:", auprc_score)


#####=====================================



    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2,  auc_score



def read_emb(fpath, dataset):
    dim = 0
    embeddings = 0
    with open(fpath) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                ll = line.split()
                assert len(ll) == 2, 'First line must be 2 numbers'
                dim = int(ll[1])
                embeddings = np.random.rand(DATASET_NUM_DIC[dataset], dim)
            else:
                line_l = line.split()
                node = line_l[0]
                emb = [float(j) for j in line_l[1:]]
                embeddings[int(node)] = np.array(emb)
    return embeddings




def logistic_embedding(k=1, dataset='bitcoin_otc', epoch=10, dirname='sgae'):
    print(epoch, dataset)
    # fpath = os.path.join(dirname, 'embedding-{}-{}-{}.npy'.format(dataset, k, epoch))
    # embeddings = np.load(fpath)

    filename = os.path.join(dirname, 'embedding-{}-{}-{}-{}.npy'.format(dataset, k, str(epoch), str('g')))
    embeddings_g = np.load(filename)

    filename = os.path.join(dirname, 'embedding-{}-{}-{}-{}.npy'.format(dataset, k, str(epoch), str('l')))
    embeddings_l = np.load(filename)

    embeddings = embeddings = np.concatenate((embeddings_g, embeddings_l), axis=1)
    # embeddings = embeddings_g.add(embeddings_l,axis=1)
    # embeddings = embeddings_l

    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = common_logistic(dataset, k, embeddings)
    return pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score


