# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 21:17:31 2020

@author: Mohammed Amine
"""

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import math



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_edges):
        super(GCN, self).__init__()
        self.fusion_layer = nn.Linear(num_edges, 1)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.LinearLayer = nn.Linear(nfeat,1)
        

    def forward(self, x, adj,args_threshold):
        adj = self.fusion_layer(adj)
        adj = torch.squeeze(adj)
        if(args_threshold == 'mean'):
            threshold = torch.mean(adj)
            adj = torch.where(adj > threshold, torch.tensor([1.0]), torch.tensor([0.0]))
                
        if(args_threshold == 'median'):
            threshold = torch.median(adj)
            adj = torch.where(adj > threshold, torch.tensor([1.0]), torch.tensor([0.0]))
            
            
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.log_softmax(x, dim=1)
        x = self.LinearLayer(torch.transpose(x,0,1))
        x = torch.transpose(x,0,1)
        return x
    
    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        
        return F.cross_entropy(pred, label, reduction='mean')