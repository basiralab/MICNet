# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:38:27 2020

@author: Mohamed
"""

import numpy as np
import torch
import pickle
import random
from graph_sampler import GraphSampler

from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def prepare_data_kfold(graphs, args, val_idx ):
    random.seed(4)
    random.shuffle(graphs)
    train_graphs=[]
    val_graphs=[]
    cv_number=args.cv_number
    
    if cv_number<len(graphs):
        val_size = len(graphs) // cv_number
        
        if(val_idx==0):
            train_graphs = graphs[(val_idx+1)*val_size:]
            val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
        elif(val_idx==cv_number-1):
            train_graphs = graphs[:val_idx*val_size]
            val_graphs = graphs[val_idx*val_size:]
        else:
            train_graphs = graphs[:val_idx*val_size]
            train_graphs = train_graphs + graphs[(val_idx+1)*val_size:]
            val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    else:
        print("Validation Method : leave one out")
        train_graphs = graphs[:val_idx] + graphs[val_idx+1:]
        val_graphs.append(graphs[val_idx])
            
    print('Number of graphs: ', len(graphs),
          '; Num training graphs: ', len(train_graphs), 
          '; Num test graphs: ', len(val_graphs))
    
    # minibatch
    dataset_sampler = GraphSampler(train_graphs)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True)

    dataset_sampler = GraphSampler(val_graphs)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False)
    positive_graphs =[]
    negative_graphs =[]
    for i in range(len(train_graphs)):
        mat_i = train_graphs[i]['adj']
        if args.IdentityMatrix == True:
            mat_i = mat_i[:,:,:mat_i.shape[2]-1]
        if train_graphs[i]['label']==0:
            negative_graphs.append(mat_i)
        else :
            positive_graphs.append(mat_i)
    
    positive_graphs_stacked = np.stack(positive_graphs, axis=0)
    positive_graphs_torch = torch.from_numpy(positive_graphs_stacked)
    
    negative_graphs_stacked = np.stack(negative_graphs, axis=0)
    negative_graphs_torch = torch.from_numpy(negative_graphs_stacked)
    
    return train_dataset_loader, val_dataset_loader, positive_graphs_torch, negative_graphs_torch


def count_populations(train_dataset_loader):
    idx_pos = 0
    idx_neg = 0
    for batch_idx, data in enumerate(train_dataset_loader):
        label = Variable(data['label'].long()).to(device)
        if label.item()==1:
            idx_pos+=1
        else:
            idx_neg+=1
    return idx_pos, idx_neg
