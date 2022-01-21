# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:59:04 2021

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
import json
from torch_geometric.data import Dataset, DataLoader
from common import DATASET_NUM_DIC
#
from fea_extra import FeaExtra
from logistic_function import logistic_embedding
from model_m import decoder,MeanAggregator_e,MeanAggregator_f,AttentionAggregator_e,AttentionAggregator_f,Encoder1, Encoder2,molecular_emb

from argparse import ArgumentParser, Namespace
from load_data_m import load_data2,metric,TestbedDataset

def parse_args():
    parser = ArgumentParser()


    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default='cuda', help='Devices')
    parser.add_argument('--seed', type=int, default=13, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dataset', default='DDI', help='Dataset')
    parser.add_argument('--f_dim', type=int, default=128, help='feature dimension')    ##1935 node_number ###128 local_emb as input to SDGNN

    parser.add_argument('--h1_dim', type=int, default=512, help='h1 Embedding dimension')

    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=300, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout k')
    parser.add_argument('--k', default=1, help='Folder k')
    parser.add_argument('--agg_f', default='mean', choices=['mean', 'attantion'], help='Aggregator choose')
    parser.add_argument('--agg_e', default='mean', choices=['mean', 'attantion'], help='Aggregator choose')

    args = parser.parse_args()

    return args

args = parse_args()
    
OUTPUT_DIR = f'./embeddings/sdgnn-{args.agg_e}'
if not os.path.exists('embeddings'):
    os.mkdir('embeddings')

if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

NEG_LOSS_RATIO = 1
INTERVAL_PRINT = 1

NUM_NODE = DATASET_NUM_DIC[args.dataset]
WEIGHT_DECAY = args.weight_decay
NODE_FEAT_SIZE = args.f_dim
H1_SIZE = args.h1_dim
EMBEDDING_SIZE1 = args.emb_dim
DEVICES = torch.device(args.devices)
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
DROUPOUT = args.dropout
K = args.k


class SDGNN(nn.Module):

    def __init__(self, enc1,enc2):
        super(SDGNN, self).__init__()
        self.enc_local = enc2

        self.enc_global = enc1

        # if args.agg == 'mean':
        #     self.aggregator = MeanAggregator
        # else:
        #     self.aggregator = AttentionAggregator
        #
        # self.encoder = Encoder

        self.score_function1 = nn.Sequential(
            nn.Linear(args.emb_dim, 1),
            nn.Sigmoid()
        )
        self.score_function2 = nn.Sequential(
            nn.Linear(args.emb_dim, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(args.emb_dim * 2, 1)

    def forward(self,  nodes, local_features,data_loader):
        embeds_global = self.enc_global(nodes,local_features)
        embeds_local = torch.Tensor().cuda()
        for batch_idx, data in enumerate(data_loader):
            data = data.to(DEVICES)
            embeds = self.enc_local(data)
            embeds_local = torch.cat((embeds_local, embeds),0)


        return embeds_global, embeds_local

    def loss(self, local_features,nodes, pos_neighbors, neg_neighbors, adj_lists1_1, adj_lists2_1, weight_dict,drug_features):
        pos_neighbors_list = [set.union(pos_neighbors[i]) for i in nodes]
        neg_neighbors_list = [set.union(neg_neighbors[i]) for i in nodes]
        unique_nodes_list = list(set.union(*pos_neighbors_list).union(*neg_neighbors_list).union(nodes))
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}

        nodes_embs = self.enc_global(unique_nodes_list,local_features)   ###global embdding

        neighbors_data = TestbedDataset(unique_nodes_list, drug_features)
        loss_loader = DataLoader(neighbors_data, batch_size=BATCH_SIZE, shuffle=False)

#        nodes_embs_local = self.enc_local(unique_nodes_list)
        nodes_embs_local = torch.Tensor().cuda()

        for batch_idx, data in enumerate(loss_loader):
            data = data.to(DEVICES)
            embeds = self.enc_local(data)
            nodes_embs_local = torch.cat((nodes_embs_local, embeds),0)


        loss_total = 0
        for index, node in enumerate(nodes):
            z1 = nodes_embs[unique_nodes_dict[node], :]
            z1_l = nodes_embs_local[unique_nodes_dict[node], :]
            pos_neigs = list([unique_nodes_dict[i] for i in pos_neighbors[node]])
            neg_neigs = list([unique_nodes_dict[i] for i in neg_neighbors[node]])
            pos_num = len(pos_neigs)
            neg_num = len(neg_neigs)

            sta_pos_neighs = list([unique_nodes_dict[i] for i in adj_lists1_1[node]])
            sta_neg_neighs = list([unique_nodes_dict[i] for i in adj_lists2_1[node]])

            pos_neigs_weight = torch.FloatTensor([weight_dict[node][i] for i in adj_lists1_1[node]]).to(DEVICES)
            neg_neigs_weight = torch.FloatTensor([weight_dict[node][i] for i in adj_lists2_1[node]]).to(DEVICES)

            if pos_num > 0:
                pos_neig_embs = nodes_embs[pos_neigs, :]
                loss_pku = F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [pos_neig_embs, z1]),
                                                              torch.ones(pos_num).to(DEVICES))
                pos_neig_embs_l = nodes_embs[pos_neigs, :]
                loss_pku += F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [pos_neig_embs_l, z1_l]),
                                                              torch.ones(pos_num).to(DEVICES))

                if len(sta_pos_neighs) > 0:
                    sta_pos_neig_embs = nodes_embs[sta_pos_neighs, :]

                    z11 = z1.repeat(len(sta_pos_neighs), 1)
                    rs = self.fc(torch.cat([z11, sta_pos_neig_embs], 1)).squeeze(-1)
                    loss_pku += F.binary_cross_entropy_with_logits(rs, torch.ones(len(sta_pos_neighs)).to(DEVICES), \
                                                                   weight=pos_neigs_weight
                                                                   )
                    s1 = self.score_function1(z1).repeat(len(sta_pos_neighs), 1)
                    s2 = self.score_function2(sta_pos_neig_embs)

                    q = torch.where((s1 - s2) > -0.5, torch.Tensor([-0.5]).repeat(s1.shape).to(DEVICES), s1 - s2)
                    tmp = (q - (s1 - s2))
                    loss_pku += 5 * torch.einsum("ij,ij->", [tmp, tmp])

                loss_total += loss_pku

            if neg_num > 0:
                neg_neig_embs = nodes_embs[neg_neigs, :]
                loss_pku = F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [neg_neig_embs, z1]),
                                                              torch.zeros(neg_num).to(DEVICES))

                neg_neig_embs_l = nodes_embs_local[neg_neigs, :]
                loss_pku = F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [neg_neig_embs_l, z1_l]),
                                                              torch.zeros(neg_num).to(DEVICES))
                if len(sta_neg_neighs) > 0:
                    sta_neg_neig_embs = nodes_embs[sta_neg_neighs, :]

                    z12 = z1.repeat(len(sta_neg_neighs), 1)
                    rs = self.fc(torch.cat([z12, sta_neg_neig_embs], 1)).squeeze(-1)

                    loss_pku += F.binary_cross_entropy_with_logits(rs, torch.zeros(len(sta_neg_neighs)).to(DEVICES), \
                                                                   weight=neg_neigs_weight)

                    s1 = self.score_function1(z1).repeat(len(sta_neg_neighs), 1)
                    s2 = self.score_function2(sta_neg_neig_embs)

                    q = torch.where(s1 - s2 > 0.5, s1 - s2, torch.Tensor([0.5]).repeat(s1.shape).to(DEVICES))

                    tmp = (q - (s1 - s2))
                    loss_pku += 5 * torch.einsum("ij,ij->", [tmp, tmp])

                loss_total += loss_pku

        return loss_total




def run(dataset, k):
    num_nodes = DATASET_NUM_DIC[dataset]     # + 3

    ####molecular

    fpath = '/home/disk2/fengyuehua/PycharmProjects/paper3_sign_directed/data/'
    drug_features = json.load(open(fpath + "drug_graph.json"))
    drug_features = pd.DataFrame(drug_features)
    # c_size = drug_features.loc['c_size']
    # edge_index = drug_features.loc['edge_index']

    #
    # del_list = []
    # for i in range(len(c_size)):
    #     if c_size[i] == [1]:
    #         drug_features = drug_features.drop(str(i), axis=1)
    #         del_list.append(i)
    # del_list = []
    # for i in range(len(c_size)):
    #     if len(edge_index[i][0]) == 0:
    #         drug_features = drug_features.drop(str(i), axis=1)
    #         del_list.append


    # # drug1_feature
    # c_size = drug_features.loc['c_size']
    # features = drug_features.loc['features']
    # print(features.size())
    # edge_index = drug_features.loc['edge_index']
    # print(edge_index.size())


    # adj_lists1, adj_lists2, adj_lists3 = load_data(k, dataset)
    # filename = './experiment-data/{}-train-{}.edgelist'.format(dataset, k)

    filename =  '/home/disk2/fengyuehua/PycharmProjects/paper3_sign_directed/data/signed_directed_adj_modified(increase and decreasel_1935).csv' ### signed_directed_adj_modified(increase and decreasel).csv
    adj = pd.read_csv(filename,index_col=0, header=0,names=[i for i in range(1935)])  # 这里不重新指定就变成字符串
    # adj.drop(1685,axis = 0)
    # adj.drop(1685,axis = 1)
    # adj.to_csv(filename)
    # for i in range(len(del_list)):
    #     adj.drop(del_list[i],axis = 0,inplace = True)
    #     adj.drop(del_list[i],axis = 1,inplace = True)
    # adj.to_csv(filename)

    innerproduct = torch.nn.Parameter(torch.Tensor(EMBEDDING_SIZE1, EMBEDDING_SIZE1))






    adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3 = load_data2(filename)

    print(k, dataset, 'data load!')

    ## ===training feature=================
    # features = nn.Embedding(num_nodes, NODE_FEAT_SIZE)
    # features.weight.requires_grad = True
    # features.weight.requires_grad = False
    ## ===one-hot feature=================
    # features1 = np.eye(NUM_NODE)
    # features1 = torch.from_numpy(features1)
    # features = features1
    # features = features.to(DEVICES)

    ## ===local embedding feature=================
    # dirname = '/home/disk2/fengyuehua/PycharmProjects/paper3_sign_directed/SDGAT_feature/embeddings/sdgnn-mean1'
    #
    # filename = os.path.join(dirname, 'embedding-{}-{}-{}-{}.npy'.format(str('DDI'), str('1'), str(200), str('l')))
    # embeddings_l = np.load(filename)
    # features = embeddings_l
    # features = torch.from_numpy(features).to(DEVICES)
    def read_local_emb(epoch):
        num = epoch -1
        dirname = '/home/disk2/fengyuehua/PycharmProjects/paper3_sign_directed/SDGAT_one_hot_feature/embeddings/sdgnn-mean'
        filename = os.path.join(dirname, 'embedding-{}-{}-{}-{}.npy'.format(str('DDI'), str('1'), str(num), str('l')))
        local_emb = np.load(filename)
        return  local_emb

    ##============================================

    adj_lists = [adj_lists1_1, adj_lists1_2,  adj_lists2_1, adj_lists2_2]
    print('adj_lists load complement')

  
    weight_dict = defaultdict(dict)
    fea_model = FeaExtra(dataset=dataset, k=k)

    for i in adj_lists1_1:
        for j in adj_lists1_1[i]:
            v_list1 = fea_model.feature_part2(i, j)
            mask = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
            counts1 = np.dot(v_list1, mask)
            weight_dict[i][j] = counts1

    for i in adj_lists2_1:
        for j in adj_lists2_1[i]:
            v_list1 = fea_model.feature_part2(i, j)
            mask = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]

            counts1 = np.dot(v_list1, mask)
            weight_dict[i][j] = counts1

    adj_lists = adj_lists
    print(len(adj_lists), 'motifs')
    
    def func(adj_list):
        edges = []
        for a in adj_list:
            for b in adj_list[a]:
                edges.append((a, b))
        edges = np.array(edges)
        num_nodes = DATASET_NUM_DIC[args.dataset]
        adj = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
        return adj





    if args.agg_f == 'mean':
        aggregator_f = MeanAggregator_f
    else:
        aggregator_f = AttentionAggregator_f

    if args.agg_e == 'mean':
        aggregator_e = MeanAggregator_e
    else:
        aggregator_e = AttentionAggregator_e
        
    adj_lists = list(map(func, adj_lists)) 

    ##molecular GCN initialize
    enc_mol = molecular_emb(num_features_xd=78,  output_dimd=128, dropoutc=0.2).to(DEVICES)

    aggs = [aggregator_f(NODE_FEAT_SIZE, H1_SIZE, DROUPOUT, num_nodes) for adj in adj_lists]
    enc1 = Encoder1(NODE_FEAT_SIZE, H1_SIZE, adj_lists, DEVICES,aggs)
    enc1 = enc1.to(DEVICES)


    aggs2 = [aggregator_e(lambda n,m: enc1(n,m), H1_SIZE, EMBEDDING_SIZE1, DROUPOUT, num_nodes) for _ in adj_lists]
    enc2 = Encoder2(lambda n,m: enc1(n,m), H1_SIZE, EMBEDDING_SIZE1, adj_lists,DEVICES, aggs2)

    decoder_D = decoder(EMBEDDING_SIZE1).to(DEVICES)
    
    model = SDGNN(enc2,enc_mol)
    model = model.to(DEVICES)

    print(model.train())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,  list(model.parameters()) + list(decoder_D.parameters()) + list(enc1.parameters()) + list(enc_mol.parameters())),  lr=LEARNING_RATE,  weight_decay=WEIGHT_DECAY )  ##+ list(features.parameters())),


    for epoch in range(1,EPOCHS + 2):
        total_loss = []
        if epoch % INTERVAL_PRINT == 0:
            model.eval()
            all_embedding_g = np.zeros((NUM_NODE, EMBEDDING_SIZE1))
            all_embedding_l = np.zeros((NUM_NODE, EMBEDDING_SIZE1))
            for i in range(0, NUM_NODE, BATCH_SIZE):
                begin_index = i
                end_index = i + BATCH_SIZE if i + BATCH_SIZE < NUM_NODE else NUM_NODE
                values = np.arange(begin_index, end_index)

                train_data = TestbedDataset(values.tolist(), drug_features)
                train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

                local_features = read_local_emb(epoch)

                embed_global, embed_local = model.forward(values.tolist(),local_features,train_loader)
                embed_global = embed_global.data.cpu().numpy()
                embed_local =embed_local.data.cpu().numpy()
                all_embedding_g[begin_index: end_index] = embed_global
                all_embedding_l[begin_index: end_index] = embed_local

            fpath = os.path.join(OUTPUT_DIR, 'embedding-{}-{}-{}-{}.npy'.format(dataset, k, str(epoch),str('g')))
            np.save(fpath, all_embedding_g)
            fpath = os.path.join(OUTPUT_DIR, 'embedding-{}-{}-{}-{}.npy'.format(dataset, k, str(epoch),str('l')))
            np.save(fpath, all_embedding_l)

            all_embeddings = np.concatenate((all_embedding_g, all_embedding_l), axis=1)
            # embeddings = embeddings_g.add(embeddings_l,axis=1)


#            pos_ratio, accuracy, f1, f1_score1, f1_score2, auc_score, precision, recall, auprc_score = metric(decoder_D, k=k, dataset=dataset, embeddings=all_embeddings)


            pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = logistic_embedding(k=k, dataset=dataset,
                                                                                                epoch=epoch,
                                                                                               dirname=OUTPUT_DIR)
            model.train()

        time1 = time.time()
        nodes_pku = np.random.permutation(NUM_NODE).tolist()
        for batch in range(NUM_NODE // BATCH_SIZE):
            optimizer.zero_grad()
            b_index = batch * BATCH_SIZE
            e_index = (batch + 1) * BATCH_SIZE
            nodes = nodes_pku[b_index:e_index]
            local_features = read_local_emb(epoch)

            loss = model.loss(local_features,
                nodes,  adj_lists1, adj_lists2, adj_lists1_1, adj_lists2_1, weight_dict,
            drug_features)
            total_loss.append(loss.data.cpu().numpy())

            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {np.mean(total_loss)}, time: {time.time()-time1}')

    ##test performance


def main():
    print('NUM_NODE', NUM_NODE)
    print('WEIGHT_DECAY', WEIGHT_DECAY)
    print('NODE_FEAT_SIZE', NODE_FEAT_SIZE)
    print('EMBEDDING_SIZE1', EMBEDDING_SIZE1)
    print('LEARNING_RATE', LEARNING_RATE)
    print('BATCH_SIZE', BATCH_SIZE)
    print('EPOCHS', EPOCHS)
    print('DROUPOUT', DROUPOUT)
    dataset = args.dataset
    run(dataset=dataset, k=K)


if __name__ == "__main__":
    main()

