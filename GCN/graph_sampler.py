# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:40:01 2020

@author: Mohamed
"""

import numpy as np
import torch
import torch.utils.data

class GraphSampler(torch.utils.data.Dataset):
    
    def __init__(self, G_list):
        self.adj_all = []
        self.label_all = []
        self.id_all = []
        
        for i in range(len(G_list)):
            self.adj_all.append(G_list[i]['adj'])
            self.label_all.append(G_list[i]['label'])
            self.id_all.append(G_list[i]['id'])

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        return {'adj':self.adj_all[idx],
                'label':self.label_all[idx],
                'id':self.id_all[idx]}
    
'''
batch_num_nodes=np.array([CBT_subject.shape[0]])
h0 = np.identity(CBT_subject.shape[0])
assign_input = np.identity(CBT_subject.shape[0])
'''