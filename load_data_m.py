# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:40:31 2021

@author: A
"""


import os
import sys
import time
import math
import random
import subprocess
import logging
import argparse
import pandas as pd
from collections import defaultdict
import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

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
#
from fea_extra import FeaExtra
from logistic_function import logistic_embedding

from argparse import ArgumentParser, Namespace


from torch_geometric.data import Dataset, DataLoader
from torch_geometric import data as DATA


class TestbedDataset(Dataset):
    def __init__(self, drug_id, drug_features):
        super(TestbedDataset, self).__init__()
        self.drug_id = drug_id
        self.drug_features = drug_features


    def len(self):
        return len(self.drug_id)

    def get(self,idx):

        #drug_feature
        id = self.drug_id[idx]
        # print(id)
        c_size=self.drug_features.loc[ 'c_size',str(id)]
        features=self.drug_features.loc['features',str(id)]
        features = torch.tensor(features[0],dtype=torch.float)
        #print(features1.size())
        edge_index=self.drug_features.loc['edge_index',str(id)]
        edge_index = torch.tensor(edge_index[0],dtype=torch.long)
        #print(edge_index1.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        Data = DATA.Data(x=features,edge_index=edge_index.transpose(1, 0))
        # Data.y = torch.tensor([float(syn)], dtype=torch.float)  # regress

        Data.__setitem__('c_size', torch.tensor([c_size],dtype=torch.long))
        return Data

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def load_data2(filename=''):


###=======train and test dataset split and save=================================
    ###read adj
    # filename =  '/home/disk2/fengyuehua/PycharmProjects/paper3_sign_directed/data/signed_directed_adj_del_double_edges(benifial and harmful)_index_num.csv' ### signed_directed_adj_modified(increase and decreasel).csv
    filename = '/home/disk2/fengyuehua/PycharmProjects/paper3_sign_directed/data/signed_directed_adj_modified(increase and decreasel_1935).csv'  ### signed_directed_adj_modified(increase and decreasel).csv


#    adj = pd.read_csv(open(filename),index_col=0)  ###filename路径中还有中文名字的时候需要加open（）
    adj = pd.read_csv(filename,index_col=0, header=0,names=[i for i in range(1935)])  # 这里不重新指定就变成字符串

#    num_drugs = 1974
#    sample_ratio = 1
#    drug_index = np.arange(0, num_drugs)
#    np.random.shuffle(drug_index)
#    drug_index = drug_index[0:int(sample_ratio * num_drugs)]
    
    test_size = 0.2
    val_size = 0.05

    adj_mats = sp.csr_matrix(adj)
    edges_all, values, shape = sparse_to_tuple(adj_mats)
    num_test = max(50, int (np.floor(len(edges_all)) * test_size))
    num_val = max(50, int (np.floor(len(edges_all)) * val_size))



    all_edge_idx = list(range(len(edges_all)))
    np.random.shuffle(all_edge_idx)

    test_edge_idx = all_edge_idx[: num_test]
    test_edges = edges_all[test_edge_idx]

    val_edge_idx = all_edge_idx[num_test: num_val+num_test]
    val_edges = edges_all[val_edge_idx]

    train_edge_idx = all_edge_idx[num_test+num_val:]
    train_edges = edges_all[train_edge_idx]   #    train_edges = np.delete(edges_all, test_edge_idx, axis=0)

    data_train = values[train_edge_idx]
    adj_train = sp.csr_matrix(
            (data_train, (train_edges[:, 0], train_edges[:, 1])),
            shape=adj_mats.shape)
    adj_train = adj_train.todense()


    data_val = values[val_edge_idx]
    adj_val = sp.csr_matrix(
        (data_val, (val_edges[:, 0], val_edges[:, 1])),
        shape=adj_mats.shape)
    adj_val = adj_val.todense()
    
    data_test = values[test_edge_idx]
    adj_test = sp.csr_matrix(
            (data_test, (test_edges[:, 0], test_edges[:, 1])),
            shape=adj_mats.shape)
    adj_test = adj_test.todense()
        
    ###save train edges and save train adj 
#    G = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph())  # 有向图
    G_train = nx.from_numpy_matrix(adj_train,create_using=nx.DiGraph())
    nx.write_weighted_edgelist(G_train,"./experiment-data/DDI-train-1.edgelist")  # 有向有权生成edgelist

    G_test = nx.from_numpy_matrix(adj_test,create_using=nx.DiGraph())
    nx.write_weighted_edgelist(G_test,"./experiment-data/DDI-test-1.edgelist")  # 有向有权生成edgelist

    G_val = nx.from_numpy_matrix(adj_val, create_using=nx.DiGraph())
    nx.write_weighted_edgelist(G_val, "./experiment-data/DDI-val-1.edgelist")  # 有向有权生成edgelist

    ###save train_adj deleted the test edges
    adj_train = pd.DataFrame(adj_train)
    filename = '/home/disk2/fengyuehua/PycharmProjects/paper3_sign_directed/code/other model/SiGAT-master/experiment-data/train_adj_benif_harm.csv'

    adj_train.to_csv(filename)

###=====================split end==============================================


    ##=======trasform adj_matrix to triple of edges
#    edges = {}
#    for i in range(adj.shape[0]):
#        for j in range(adj.shape[1]):
#            if adj.iloc[i, j] != 0:
#                edges[(i, j)] = adj.iloc[i, j]

#    G = nx.Graph()
#    for key, value in edges.items():
#        G.add_edges_from([key], relationship=value)  ##NOTICE key must [key]   ##   label=value      ##也可以从邻接表直接构建  正负看做是边的权重 
        


    # filename = 'E:/paper3_sign_directed/code/other model/SiGAT-master/experiment-data/train_adj_benif_harm.csv'
    adj = pd.read_csv(filename,index_col=0, header=0,names=[i for i in range(1935)])  # 这里不重新指定就变成字符串
    G = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph())  # 有向图
    edges = list(nx.to_edgelist(G))
    
    adj_lists1 = defaultdict(set)
    adj_lists1_1 = defaultdict(set)
    adj_lists1_2 = defaultdict(set)
    adj_lists2 = defaultdict(set)
    adj_lists2_1 = defaultdict(set)
    adj_lists2_2 = defaultdict(set)
    adj_lists3 = defaultdict(set)    
    
    for i in range(len(edges)):
        drug1 = int(edges[i][0])
        drug2 = int(edges[i][1])
        v = int(edges[i][2]['weight'])
        adj_lists3[drug2].add(drug1)  ##adj_lists3没有方向，只有正负
        adj_lists3[drug1].add(drug2)

        if v == 1:
            adj_lists1[drug1].add(drug2)  ##adj_lists1没有方向，没有正负
            adj_lists1[drug2].add(drug1)

            adj_lists1_1[drug1].add(drug2)  ###adj_lists1_1只有出的方向：source->target
            adj_lists1_2[drug2].add(drug1)  ###adj_lists1_2只有入的方向：target->source
        else:
            adj_lists2[drug1].add(drug2)  ##adj_lists2==adj_lists1
            adj_lists2[drug2].add(drug1)

            adj_lists2_1[drug1].add(drug2)
            adj_lists2_2[drug2].add(drug1)


    # with open(filename) as fp:
    #     for i, line in enumerate(fp):
    #         info = line.strip().split()
    #         person1 = int(info[0])
    #         person2 = int(info[1])
    #         v = int(info[2])
    #         adj_lists3[person2].add(person1)   ##adj_lists3没有方向，只有正负
    #         adj_lists3[person1].add(person2)
    #
    #         if v == 1:
    #             adj_lists1[person1].add(person2)   ##adj_lists1没有方向，没有正负
    #             adj_lists1[person2].add(person1)
    #
    #             adj_lists1_1[person1].add(person2)  ###adj_lists1_1只有出的方向：source->target
    #             adj_lists1_2[person2].add(person1)  ###adj_lists1_2只有入的方向：target->source
    #         else:
    #             adj_lists2[person1].add(person2)     ##adj_lists2==adj_lists1
    #             adj_lists2[person2].add(person1)
    #
    #             adj_lists2_1[person1].add(person2)
    #             adj_lists2_2[person2].add(person1)

    return adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3


def read_val_data(dataset, k):

    val_X = []
    val_Y = []
    # with open('./experiment-data/{}-val-{}.edgelist'.format(dataset, k)) as f:
    #     for line in f:
    #         i, j, _ = line.split()
    #         i = int(i)
    #         j = int(j)
    #         flag_p = 1
    #         flag_n = 0
    #         val_X.append((i, j))
    #         if i > j:
    #             val_Y.append(flag_n)
    #         else:
    #             val_Y.append(flag_p)
    #
    with open('./experiment-data/{}-test-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1)/2)
            val_X.append((i, j))
            val_Y.append(flag)

    return np.array(val_X), np.array(val_Y)


def read_test_data(dataset, k):


    test_X = []
    test_Y = []
    with open('./experiment-data/{}-test-{}.edgelist'.format(dataset, k)) as f:
        for line in f:
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag_p = 1
            flag_n = 0
            test_X.append((i, j))
            if i > j:
                test_Y.append(flag_n)
            else:
                test_Y.append(flag_p)
    return np.array(test_X), np.array(test_Y)


def read_emb(num_nodes, fpath):
    dim = 0
    embeddings = 0
    with open(fpath) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                dim = int(line.split()[1])
                embeddings = np.random.rand(num_nodes, dim)
            else:
                line_l = line.split()
                node = line_l[0]
                emb = [float(j) for j in line_l[1:]]
                assert len(emb) == dim
                embeddings[int(node)] = np.array(emb)
    return embeddings




def metric(decoder_D,dataset, k, embeddings):
    val_X, val_Y = read_val_data(dataset, k )

    pred_score = []
    def sigmoid(x):
        y = 1/(1+np.exp(-x))
        return y

    ##E*D*ET  instead  inner product



    for i, j in val_X:
        #pred_score.append(sigmoid(np.dot(embeddings[i], embeddings[j])) )##torch.nn.ReLU()

        pred_score.append(decoder_D(embeddings[i], embeddings[j]))
    labels_all = val_Y
    preds_all = pred_score
    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    # aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    precision,recall,pr_thresholds = precision_recall_curve(labels_all,preds_all)
    auprc_score = auc(recall,precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(labels_all, preds_all)
    auc_score = auc(fpr, tpr)
    predicted_score = np.zeros(shape=(len(labels_all), 1))
    predicted_score[preds_all > threshold] = 1
    confusion_matri = confusion_matrix(y_true=labels_all, y_pred=predicted_score)
    # print("confusion_matrix:", confusion_matri)
    f1 = f1_score(labels_all, predicted_score)
    accuracy = accuracy_score(labels_all,predicted_score)
    precision = precision_score(labels_all,predicted_score)
    recall = recall_score(labels_all,predicted_score)

    pos_ratio = np.sum(val_Y) / val_Y.shape[0]
    f1_score1 = metrics.f1_score(val_Y, predicted_score, average='macro')
    f1_score2 = metrics.f1_score(val_Y, predicted_score, average='micro')


    print("pos_ratio:", pos_ratio)
    print('accuracy:', accuracy)
    print("f1_score:", f1)
    print("macro f1_score:", f1_score1)
    print("micro f1_score:", f1_score2)
    print("auc score:", auc_score)
    print("precision score:", precision)
    print("recall score:", recall)
    print("pr score:", auprc_score)

    return   pos_ratio, accuracy, f1, f1_score1, f1_score2, auc_score ,precision, recall,auprc_score ###              roc_sc, auprc_score,accuracy,precision,recall,f




