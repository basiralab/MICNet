# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 02:27:47 2020

@author: Mohamed
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PositivePopulationFusion(nn.Module):
    def __init__(self, num_subjects):
        super(PositivePopulationFusion, self).__init__()
        self.num_subjects = num_subjects
        self.weight = nn.Parameter(torch.randn(num_subjects, 1).to(device))
    def forward(self, H_population):
        s_weight = self.weight
        norm_weight = torch.div(s_weight,torch.sum(s_weight,0).item())
        H_fusion = torch.squeeze(torch.matmul(H_population , norm_weight))
        H_fusion = (H_fusion + torch.t(H_fusion))/2
        # Uncomment the next line if needed to normalize the fusion output
        #H_fusion =  torch.div(H_fusion-torch.min(H_fusion),torch.max(H_fusion)-torch.min(H_fusion))
        return H_fusion
    
    
class NegativePopulationFusion(nn.Module):
    def __init__(self, num_subjects):
        super(NegativePopulationFusion, self).__init__()
        self.num_subjects = num_subjects
        self.weight = nn.Parameter(torch.randn(num_subjects, 1).to(device))
    def forward(self, H_population):
        s_weight = self.weight
        norm_weight = torch.div(s_weight,torch.sum(s_weight,0).item())
        H_fusion = torch.squeeze(torch.matmul(H_population , norm_weight))
        H_fusion = (H_fusion + torch.t(H_fusion))/2
        # Uncomment the next line if needed to normalize the fusion output
        #H_fusion =  torch.div(H_fusion-torch.min(H_fusion),torch.max(H_fusion)-torch.min(H_fusion))
        return H_fusion

class LearnableDistance(nn.Module):
    def __init__(self, num_nodes):
        super(LearnableDistance, self).__init__()
        self.num_nodes = num_nodes
        self.weight_Pos = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device))
        self.weight_Neg = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device))
    def forward(self, H, H_pos, H_neg):
        pos_weight = self.weight_Pos
        neg_weight = self.weight_Neg
        norm_pos_weight = torch.div(pos_weight-torch.min(pos_weight),torch.max(pos_weight)-torch.min(pos_weight))
        norm_neg_weight = torch.div(neg_weight-torch.min(neg_weight),torch.max(neg_weight)-torch.min(neg_weight))
        H_pos = torch.mul(H_pos, norm_pos_weight)
        H_neg = torch.mul(H_neg, norm_neg_weight)
        distance_positive = torch.unsqueeze(torch.dist(H, H_pos, 2),0)
        distance_negative = torch.unsqueeze(torch.dist(H, H_neg, 2),0)
        distances = torch.cat([distance_negative, distance_positive])
        distances = torch.unsqueeze(distances,0)
        return distances 

class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, num_layers,norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.drop_out = nn.Dropout()
        
    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)
        deg = torch.sum(H, dim=1) #to check 
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    def forward(self, A):
        A = A.permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        #H = self.drop_out(H)
        H = torch.mean(H,0)
        H = (H + torch.t(H))/2
        return H

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
