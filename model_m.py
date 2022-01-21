# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:46:19 2021

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
from common import DATASET_NUM_DIC
from fea_extra import FeaExtra
from logistic_function import logistic_embedding
from argparse import ArgumentParser, Namespace

DEVICES = torch.device('cuda')

from torch.optim import lr_scheduler
import json
from torch_geometric.nn import GCNConv, global_max_pool as gmp



# GCN based model
class molecular_emb(torch.nn.Module):
    def __init__(self, num_features_xd=78,  output_dimd=128, dropoutc=0.2):

        super(molecular_emb, self).__init__()
        # SMILES graph branch
        self.conv1 = GCNConv(num_features_xd, num_features_xd)  # 定义第一层图卷积78*78
        self.conv2= GCNConv(num_features_xd, num_features_xd*2) # 定义第二层图卷积78*156
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4) # 定义第三层图卷积156*312
        self.fc_g1= torch.nn.Linear(num_features_xd*4, 1024)  # 定义平滑处理药物嵌入特征的全连接层1，312*1024
        self.fc_g2 = torch.nn.Linear(1024, output_dimd)   #定义平滑处理药物嵌入特征的全连接层2，1024
        self.relu = nn.ReLU()
        self.dropoutc = nn.Dropout(dropoutc)

    def forward(self, data,regress=True):
        #print(data)
        #print(data0)
        # get graph input
        #x1, edge_index1, x2, edge_index2, batch = data.x, data.edge_index,data.x2.x, data.x2.edge_index, data.batch
        # get protein input

        x1, edge_index1, batch1= data['x'], data['edge_index'] ,data['batch']
        #print(batch1)
        #print(batch1)
        x1 = x1.float().cuda()
        x1 = self.conv1(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.conv3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)
        #print(x1.size)
        #x1 = gmp(x1, len(data))       # global max pooling

        # flattenv
        x1 = self.relu(self.fc_g1(x1))
        x1 = self.dropoutc(x1)
        #print(x1.size())
        x1 = self.fc_g2(x1)
        x1 = self.relu(x1)
        return x1


class decoder(nn.Module):
    def __init__(self, EMBEDDING_SIZE1):
        super(decoder, self).__init__()
        self.dim = EMBEDDING_SIZE1
        self.out_linear_layer1 = nn.Linear(self.dim*4, 64)
        self.out_linear_layer2 = nn.Linear(64, 64)
        self.out_linear_layer3 = nn.Linear(64, 64)
        self.out_linear_layer4 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

        self.d = nn.Parameter(torch.FloatTensor(self.dim*2, self.dim*2 ))
        self.r = nn.Parameter(torch.FloatTensor(self.dim*2, self.dim*2 ))

        nn.init.kaiming_normal_(self.d.data)

    def forward(self, emb1,emb2):
        emb11 = torch.Tensor(emb1).reshape((1,self.dim*2))#.to(DEVICES)
        emb22 = torch.Tensor(emb2).reshape((1,self.dim*2))#.to(DEVICES)
        #new_embeddings = self.out_linear_layer(self.features(n2))

        #out_score =  torch.FloatTensor(1).to(DEVICES)
        #inter = torch.mm(emb11, self.d)
        #out_score = F.sigmoid(torch.mm(inter,emb22.T))
        ##EDET framwork
        inter = torch.einsum("ij,jl->il", [emb11.to(DEVICES), self.d.to(DEVICES)])
        out_score = F.sigmoid(torch.einsum("ij,jl->il", [inter, emb22.T.to(DEVICES)]))   #F.leaky_relu

        #DNN framwork
        #edge = torch.cat((emb11.to(DEVICES),emb22.to(DEVICES)), dim=1)
        #out = self.relu(self.out_linear_layer1(edge))
        #out = self.relu(self.out_linear_layer2(out))
        #out = self.relu(self.out_linear_layer3(out))
        #out_score = F.sigmoid(self.out_linear_layer4(out))


        return out_score.data.cpu()

    #new_embeddings = self.out_linear_layer(self.features(n2))

    #original_node_edge = np.array([self.unique_nodes_dict[nodes], self.unique_nodes_dict[nodes]]).T
    #edges = np.vstack((edges, original_node_edge))

    #edges = torch.LongTensor(edges).to(DEVICES)

    #edge_h_2 = torch.cat((new_embeddings[edges[:, 0], :], new_embeddings[edges[:, 1], :]), dim=1)

    #edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), self.slope_ratio))
    #indices = edges

    #matrix = torch.sparse_coo_tensor(indices.t(), edges_h[:, 0], \
     #                                torch.Size([batch_node_num, batch_node_num]), device=DEVICES)
    #row_sum = torch.sparse.mm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))

    #results = torch.sparse.mm(matrix, new_embeddings)

    #
    #output_emb = results.div(row_sum)



class AttentionAggregator_f(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate, node_num, slope_ratio=0.1):
        super(AttentionAggregator_f, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.slope_ratio = slope_ratio
        self.a = nn.Parameter(torch.FloatTensor(out_dim * 2, 1))
        nn.init.kaiming_normal_(self.a.data)

        self.out_linear_layer = nn.Linear(self.in_dim, self.out_dim)
        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)

    def forward(self, nodes, adj, ind,local_features):
        node_pku = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_pku[edges[:, 0]]

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))

        batch_node_num = len(unique_nodes_list)
        # this dict can map new i to originial node id
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)

        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)
        n2 = list(n2.cpu().numpy())


        batch_features = local_features[n2]
        batch_features = torch.tensor(batch_features, dtype=torch.float32).to(DEVICES)
        new_embeddings = self.out_linear_layer(batch_features)  # self.features(n2)

        # new_embeddings = self.out_linear_layer(self.features(n2))

        original_node_edge = np.array([self.unique_nodes_dict[nodes], self.unique_nodes_dict[nodes]]).T
        edges = np.vstack((edges, original_node_edge))

        edges = torch.LongTensor(edges).to(DEVICES)

        edge_h_2 = torch.cat((new_embeddings[edges[:, 0], :], new_embeddings[edges[:, 1], :]), dim=1)

        edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), self.slope_ratio))
        indices = edges

        matrix = torch.sparse_coo_tensor(indices.t(), edges_h[:, 0], \
                                         torch.Size([batch_node_num, batch_node_num]), device=DEVICES)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))

        results = torch.sparse.mm(matrix, new_embeddings)

        output_emb = results.div(row_sum)
        return output_emb[self.unique_nodes_dict[nodes]]




class AttentionAggregator_e(nn.Module):
    def __init__(self, features, in_dim, out_dim, dropout_rate, node_num,  slope_ratio=0.1):
        super(AttentionAggregator_e, self).__init__()
        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.slope_ratio = slope_ratio
        self.a = nn.Parameter(torch.FloatTensor(out_dim * 2, 1))

        nn.init.kaiming_normal_(self.a.data)

        self.out_linear_layer = nn.Linear(self.in_dim, self.out_dim)
        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)


    def forward(self, nodes, adj, ind,local_features):
        node_pku = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_pku[edges[:, 0]]

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))

        batch_node_num = len(unique_nodes_list)
        # this dict can map new i to originial node id
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)

        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)


        # new_embeddings = self.out_linear_layer(self.features(n2))
        new_embeddings = self.out_linear_layer(self.features(n2,local_features))


        original_node_edge = np.array([self.unique_nodes_dict[nodes], self.unique_nodes_dict[nodes]]).T
        edges = np.vstack((edges, original_node_edge))

        edges = torch.LongTensor(edges).to(DEVICES)

        edge_h_2 = torch.cat((new_embeddings[edges[:, 0], :], new_embeddings[edges[:, 1], :]), dim=1)

        edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), self.slope_ratio))
        indices = edges

        matrix = torch.sparse_coo_tensor(indices.t(), edges_h[:, 0], \
                                         torch.Size([batch_node_num, batch_node_num]), device=DEVICES)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))

        results = torch.sparse.mm(matrix, new_embeddings)

        output_emb = results.div(row_sum)
        return output_emb[self.unique_nodes_dict[nodes]]

class MeanAggregator_e(nn.Module):
    def __init__(self, features, in_dim, out_dim,dropout_rate, node_num):
        super(MeanAggregator_e, self).__init__()

        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_linear_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
            nn.Linear(self.out_dim, self.out_dim)
        )

        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, nodes, adj, ind,local_features):
        """

        :param nodes:
        :param adj:
        :return:
        """
        mask = [1, 1, 0, 0]
        node_tmp = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_tmp[edges[:, 0]]

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))
        batch_node_num = len(unique_nodes_list)
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)

        ## transform 2 new axis
        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)

        # n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)
        # n2 = list(n2.cpu().numpy())
        # batch_features = features[n2]
        # batch_features= torch.tensor(batch_features, dtype=torch.float32).to(DEVICES)
        # new_embeddings = self.out_linear_layer(batch_features)   #self.features(n2)

        new_embeddings = self.out_linear_layer(self.features(n2,local_features))
        edges = torch.LongTensor(edges).to(DEVICES)

        values = torch.where(edges[:, 0] == edges[:, 1], torch.FloatTensor([mask[ind]]).to(DEVICES), torch.FloatTensor([1]).to(DEVICES))
        # values = torch.ones(edges.shape[0]).to(DEVICES)
        matrix = torch.sparse_coo_tensor(edges.t(), values, torch.Size([batch_node_num, batch_node_num]), device=DEVICES)
        row_sum = torch.spmm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(DEVICES), row_sum)

        results = torch.spmm(matrix, new_embeddings)
        output_emb = results.div(row_sum)

        return output_emb[self.unique_nodes_dict[nodes]]


class MeanAggregator_f(nn.Module):
    def __init__(self,  in_dim, out_dim,dropout_rate, node_num):
        super(MeanAggregator_f, self).__init__()

        # self.features = features   #        ## local_emb as input to SDGNN
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_linear_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
            nn.Linear(self.out_dim, self.out_dim)
        )

        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, nodes, adj, ind, local_features):
        """

        :param nodes:
        :param adj:
        :return:
        """
        mask = [1, 1, 0, 0]
        node_tmp = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_tmp[edges[:, 0]]

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))
        batch_node_num = len(unique_nodes_list)
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)

        ## transform 2 new axis
        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        # n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)

        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)
        n2 = list(n2.cpu().numpy())



        ##=================================
        # batch_features = self.feature[n2]
        batch_features = local_features[n2]
        batch_features= torch.tensor(batch_features, dtype=torch.float32).to(DEVICES)
        new_embeddings = self.out_linear_layer(batch_features)   #self.features(n2)

        # new_embeddings = self.out_linear_layer(self.features(n2))
        edges = torch.LongTensor(edges).to(DEVICES)

        values = torch.where(edges[:, 0] == edges[:, 1], torch.FloatTensor([mask[ind]]).to(DEVICES), torch.FloatTensor([1]).to(DEVICES))
        # values = torch.ones(edges.shape[0]).to(DEVICES)
        matrix = torch.sparse_coo_tensor(edges.t(), values, torch.Size([batch_node_num, batch_node_num]), device=DEVICES)
        row_sum = torch.spmm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(DEVICES), row_sum)

        results = torch.spmm(matrix, new_embeddings)
        output_emb = results.div(row_sum)

        return output_emb[self.unique_nodes_dict[nodes]]

class Encoder1(nn.Module):
    """
    Encode features to embeddings
    """

    def __init__(self,  feature_dim, embed_dim, adj_lists,DEVICES,aggs):
        super(Encoder1, self).__init__()

        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggs = aggs

        self.embed_dim = embed_dim
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(DEVICES)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        self.nonlinear_layer = nn.Sequential(
                nn.Linear(embed_dim*4+feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, embed_dim)
        )
        #
        # self.nonlinear_layer = nn.Sequential(
        #         nn.Linear((len(adj_lists) + 1) * feature_dim, feature_dim),
        #         nn.Tanh(),
        #         nn.Linear(feature_dim, embed_dim)
        # )

        self.nonlinear_layer.apply(init_weights)


    def forward(self, nodes,local_features):
        """
        Generates embeddings for nodes.
        """

        if not isinstance(nodes, list) and nodes.is_cuda:
            nodes = nodes.data.cpu().numpy().tolist()

        neigh_feats = [agg(nodes, adj, ind,local_features) for adj, agg, ind in zip(self.adj_lists, self.aggs, range(len(self.adj_lists)))]


        # n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)
        # n2 = list(n2.cpu().numpy())
        self_feats = local_features[nodes]
        self_feats = torch.tensor(self_feats, dtype=torch.float32).to(DEVICES)


        # self_feats = self.features(torch.LongTensor(nodes).to(DEVICES))
        combined = torch.cat([self_feats] + neigh_feats, 1)
        combined = self.nonlinear_layer(combined)
        return combined

        # k = self.k(self_feats)
class Encoder2(nn.Module):
    """
    Encode features to embeddings
    """

    def __init__(self, features, feature_dim, embed_dim, adj_lists,DEVICES,aggs):
        super(Encoder2, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggs = aggs

        self.embed_dim = embed_dim
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(DEVICES)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        # self.nonlinear_layer = nn.Sequential(
        #         nn.Linear((len(adj_lists) + 1) * feature_dim, feature_dim),
        #         nn.Tanh(),
        #         nn.Linear(feature_dim, embed_dim)
        # )
        self.nonlinear_layer = nn.Sequential(
                nn.Linear(feature_dim + embed_dim*4, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, embed_dim)
        )

        self.nonlinear_layer.apply(init_weights)


    def forward(self, nodes,local_features):
        """
        Generates embeddings for nodes.
        """

        if not isinstance(nodes, list) and nodes.is_cuda:
            nodes = nodes.data.cpu().numpy().tolist()

        neigh_feats = [agg(nodes, adj, ind,local_features) for adj, agg, ind in zip(self.adj_lists, self.aggs, range(len(self.adj_lists)))]
        self_feats = self.features(torch.LongTensor(nodes).to(DEVICES),local_features)
        combined = torch.cat([self_feats] + neigh_feats, 1)
        combined = self.nonlinear_layer(combined)
        return combined
