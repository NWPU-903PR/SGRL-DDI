#!/usr/bin/env python3
# -*- coding: utf-8 -*-



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



import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict
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
BATCH_SIZE = 200


###===================obtain labels of train and test sets=========================
###=======================labels of positive sign and negtive sign=================
# """
def read_train_data(dataset, k):
    train_X = []
    train_y = []
    with open('./experiment-data/{}-train-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1) / 2)  ###将标签1和-1转成了二分类1和0
            train_X.append((i, j))
            train_y.append(flag)
    return np.array(train_X), np.array(train_y)


def read_test_data(dataset, k):
    test_X = []
    test_y = []
    with open('./experiment-data/{}-test-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1) / 2)
            test_X.append((i, j))
            test_y.append(flag)
    return np.array(test_X), np.array(test_y)
#"""
###======================================================================
###=======direct sample  1  =============================================
"""
def read_train_data(dataset, k):
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
    return np.array(train_X), np.array(train_y)

def read_test_data(dataset, k):

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
    return np.array(test_X), np.array(test_y)
"""
###======================================================================
###=======direct sample  2  =============================================
###labels of positive edges of source->target  and negtive edges of source<-target (将样本分为正样本和负样本)
###依据：i->j,如果i>j:负样本    i<j:正样本
"""
def read_train_data(dataset, k):
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
    return np.array(train_X), np.array(train_y)

def read_test_data(dataset, k):

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
    return np.array(test_X), np.array(test_y)
###labels of positive edges of source->target  and negtive edges (将正样本取反，采样每条变对应的负样本)

"""


def metric(pred_p, y):
    pred_l = np.argmax(pred_p, axis=1)   ###.detach().numpy()
    test_y = y
    pred = pred_p        ##pred_p.detach().numpy()
    pos_ratio = np.sum(test_y) / test_y.shape[0]
    accuracy = metrics.accuracy_score(test_y, pred_l)
    f1_score0 = metrics.f1_score(test_y, pred_l)
    f1_score1 = metrics.f1_score(test_y, pred_l, average='macro')
    f1_score2 = metrics.f1_score(test_y, pred_l, average='micro')

    auc_score = metrics.roc_auc_score(test_y, pred[:, 1])
    print("pos_ratio:", pos_ratio)
    print('accuracy:', accuracy)
    print("f1_score:", f1_score0)
    print("macro f1_score:", f1_score1)
    print("micro f1_score:", f1_score2)
    print("auc score:", auc_score)
    ###================================
    precision, recall, pr_thresholds = precision_recall_curve(test_y, pred[:, 1])
    auprc_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(test_y, pred[:, 1])
    auc_score = auc(fpr, tpr)
    predicted_score = np.zeros(shape=(len(test_y), 1))
    predicted_score[pred[:, 1] > threshold] = 1
    confusion_matri = confusion_matrix(y_true=test_y, y_pred=predicted_score)
    # print("confusion_matrix:", confusion_matri)
    f1 = f1_score(test_y, predicted_score)
    accuracy = accuracy_score(test_y, predicted_score)
    precision = precision_score(test_y, predicted_score)
    recall = recall_score(test_y, predicted_score)
    print("new auc_score:", auc_score)
    print('new accuracy:', accuracy)
    print("new precision:", precision)
    print("new macro f1_score:", f1_score1)
    print("new recall:", recall)
    print("new f1:", f1)
    print("new auprc_score:", auprc_score)


def read_emb(epoch=10):
    print(epoch)
    # fpath = os.path.join(dirname, 'embedding-{}-{}-{}.npy'.format(dataset, k, epoch))
    # embeddings = np.load(fpath)
    dirname = '/home/disk2/fengyuehua/PycharmProjects/paper3_sign_directed/SDGAT_one_hot_feature/embeddings/sdgnn-mean'

    filename = os.path.join(dirname, 'embedding-{}-{}-{}-{}.npy'.format(str('DDI'), str('1'), str(epoch), str('g')))
    embeddings_g = np.load(filename)

    # filename = os.path.join(dirname, 'embedding-{}-{}-{}-{}.npy'.format(str('DDI'), str('1'), str(epoch), str('l')))
    # embeddings_l = np.load(filename)

    # embeddings = np.concatenate((embeddings_g, embeddings_l), axis=1)
    embeddings = embeddings_g

    # embeddings = embeddings_g.add(embeddings_l,axis=1)
    # embeddings = embeddings_l
    return embeddings



class Dataset_train(Dataset):
    def __init__(self, embeddings):
        self.train_X, self.train_y = read_train_data(dataset, k=1)

        self.train_X1 = []

        for i, j in self.train_X:
            self.train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

        self.x = self.train_X1
        self.y = self. train_y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.train_X)


class Dataset_test(Dataset):
    def __init__(self, embeddings):
        self.test_X, self.test_y = read_test_data(dataset, k=1)

        self.test_X1 = []

        for i, j in self.test_X:
            self.test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

        self.x = self.test_X1
        self.y = self.test_y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.test_X)





class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()



        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(256 * 1, 128),   ####input-dim
            torch.nn.Dropout(0.01),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.Dropout(0.01),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.01),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 32),
            torch.nn.Dropout(0.01),
            torch.nn.ReLU(),

            torch.nn.Linear(32, 2),
            torch.nn.Sigmoid(),
        )
        self.lr = 0.001
        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.dnn(x)
        return out


def DNN_train(train_loader, test_loader, epoch):
    model = DNN()
    opt = model.opt



    for i in range(epoch):
        step = 0
        ts = time.time()
        loss = 0.
        for (x, y) in train_loader:
            model.train()
            step += 1
            out = model(x)
            loss = model.loss(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # if (step % 300 == 0):
            #     print(epoch, step, loss.data.numpy())
            loss += loss
        print("epoch:{},loss:{}".format(i,loss))

    model.eval()
    #val data
    pred = torch.Tensor()
    label = torch.Tensor()

    for (test_x, test_y) in test_loader:
        test_out = model(test_x)
        pred = torch.cat((pred,test_out),0)
        label = torch.cat((label,test_y.to(torch.float32)),0)

    label = label.detach().numpy()
    pred = pred.detach().numpy()
    metric(pred, label)

    print(time.time() - ts)
##test data computing whole result




dataset = 'DDI'
#    pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = logistic_embedding9(k=1, dataset=dataset, epoch=100, dirname='sigat')

# print("pos_ratio:", pos_ratio)
# print('accuracy:', accuracy)
# print("f1_score:", f1_score0)
# print("macro f1_score:", f1_score1)
# print("micro f1_score:", f1_score2)
# print("auc score:",auc_score)

embeddings = read_emb(epoch=200)

Dataset_train = Dataset_train(embeddings)
Dataset_test = Dataset_test(embeddings)


train_loader = DataLoader(
            dataset=Dataset_train,
            batch_size=BATCH_SIZE,
            shuffle=True
             )
test_loader = DataLoader(
            dataset=Dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=True
             )



DNN_train(train_loader, test_loader, 200)




