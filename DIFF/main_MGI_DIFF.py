# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:58:29 2020

@author: Mohamed
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle
from torch.autograd import Variable
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import cross_val
from model import GTN
from model import PositivePopulationFusion
from model import NegativePopulationFusion
from model import LearnableDistance
import encoders_MGI as encoders

import time

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import mlab


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate(dataset, model_GTN, model_DIFFPOOL, args, name='Test', max_num_examples=None):
    model_GTN.eval()
    model_DIFFPOOL.eval()
    tracked_Dict = {}
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        CBT_subject = model_GTN(adj)
    
        batch_num_nodes=np.array([CBT_subject.shape[0]])
        h0 = np.identity(CBT_subject.shape[0])
        assign_input = np.identity(CBT_subject.shape[0])
        
        h0 = Variable(torch.from_numpy(h0).float(), requires_grad=False).cpu()
        assign_input = Variable(torch.from_numpy(assign_input).float(), requires_grad=False).cpu()
        
        h0 = torch.unsqueeze(h0, 0)
        CBT_subject = torch.unsqueeze(CBT_subject, 0)
        assign_input = torch.unsqueeze(assign_input, 0)
        
        ypred = model_DIFFPOOL(h0, CBT_subject, batch_num_nodes, assign_x=assign_input)

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        '''if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break'''

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    tracked_Dict['preds'] = preds
    tracked_Dict['labels'] = labels
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    return tracked_Dict

def plot_heatmap(H):
    # plot positive fusion template tensor 
    H_population = H.detach().numpy()
    mask_adj = np.zeros_like(H_population)
    mask_adj[np.triu_indices_from(mask_adj)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(30, 30))
        ax = sns.heatmap(H_population, mask=mask_adj, square=True,annot=True)

def minmax_sc(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x

def train(args, train_dataset, val_dataset, model_GTN, model_DIFFPOOL, positive_graphs_torch, negative_graphs_torch):
    Epoch_losses = []    
    GTN_losses = []
    params = list(model_GTN.parameters()) + list(model_DIFFPOOL.parameters()) 
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.00001)
    tracked_Dicts = []
    for epoch in range(args.num_epochs):
        print("Epoch ",epoch)
        model_GTN.train()
        model_DIFFPOOL.train()
        total_time = 0
        avg_loss = 0.0
        
        preds = []
        labels = []
        for batch_idx, data in enumerate(train_dataset):
            begin_time = time.time()
            
            
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            adj_id = Variable(data['id'].int()).to(device)
            
            CBT_subject = model_GTN(adj)
            # added
            batch_num_nodes=np.array([CBT_subject.shape[0]])
            h0 = np.identity(CBT_subject.shape[0])
            assign_input = np.identity(CBT_subject.shape[0])
            
            h0 = Variable(torch.from_numpy(h0).float(), requires_grad=False).cpu()
            assign_input = Variable(torch.from_numpy(assign_input).float(), requires_grad=False).cpu()
            
            h0 = torch.unsqueeze(h0, 0)
            CBT_subject = torch.unsqueeze(CBT_subject, 0)
            assign_input = torch.unsqueeze(assign_input, 0)
            
            ypred = model_DIFFPOOL(h0, CBT_subject , batch_num_nodes, assign_x=assign_input)
            
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            
            
            loss = model_DIFFPOOL.loss(ypred, label)
            
            model_GTN.zero_grad()
            model_DIFFPOOL.zero_grad()
            
            loss.backward()
            #nn.utils.clip_grad_norm_(model_DIFFPOOL.parameters(), args.clip)
            optimizer.step()
            
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        print("Train accuracy : ", np.mean( preds == labels ))
        #result_train = evaluate(train_dataset, model_GTN, model_DIFFPOOL, args, name='Train', max_num_examples=100)
        tracked_Dict = evaluate(val_dataset, model_GTN, model_DIFFPOOL, args, name='Test', max_num_examples=100)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        tracked_Dicts.append(tracked_Dict)
    return tracked_Dicts

def load_data(args):
    #Load graphs and labels
    with open('data/'+args.dataset+'/'+args.dataset+'_edges','rb') as f:
        adjacencies = pickle.load(f)        
    with open('data/'+args.dataset+'/'+args.dataset+'_labels','rb') as f:
        labels = pickle.load(f)
    #Normalize inputs
    for subject in range(len(adjacencies)):
        for view in range(adjacencies[0].shape[2]):
            adjacencies[subject][:,:,view] = minmax_sc(adjacencies[subject][:,:,view])
    #Add Id view in every subject
    num_nodes = adjacencies[subject].shape[0]
    if args.IdentityMatrix==True:
        I = np.identity(num_nodes)
        IdM = np.expand_dims(I, axis=2)
        for subject in range(len(adjacencies)):
            adjacencies[subject] = np.concatenate([adjacencies[subject], IdM], axis =2) 
    #Create List of Dictionaries
    G_list=[]
    for i in range(len(labels)):
        G_element = {"adj":   adjacencies[i],"label": labels[i],"id":  i,}
        G_list.append(G_element)
    return G_list

def arg_parse():
    parser = argparse.ArgumentParser(description='Graph Classification')
    parser.add_argument('--dataset', type=str, default='RH_ASDNC_extracted',
                        help='Dataset')
    parser.add_argument('--fusion_method', type=str, default='MGI',
                        help='Fusion Method')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Training Epochs')
    parser.add_argument('--num_layers', type=int, default=3, #2
                        help='number of layer')
    parser.add_argument('--batch-size', type=int, default=1, # only works with batchsize=1
                        help='Batch size.')
    parser.add_argument('--cv_number', type=int, default=4,
                        help='number of validation folds.')
    parser.add_argument('--lambda_GTN', type=float, default=1,
                        help='multiplication term for loss.')
    parser.add_argument('--lambda_Fusion', type=float, default=0.1,
                        help='multiplication term for loss.')
    parser.add_argument('--IdentityMatrix', default=False, action='store_true',
                        help='Add Identity matrix View in adjacency')
    ##################
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int, default=64,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=2,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--assign-ratio', dest='assign_ratio', type=float, default=0.1,
                        help='ratio of number of nodes in consecutive layers')
    parser.add_argument('--num-pool', dest='num_pool', type=int, default=1,
                        help='number of pooling layers')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0,
                        help='Dropout rate.')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--clip', dest='clip', type=float, default=2.0,
            help='Gradient clipping.')
    
    
    
    return parser.parse_args()

def benchmark_task(args):
    G_list = load_data(args)
    num_edge = G_list[0]['adj'].shape[-1]
    num_nodes = G_list[0]['adj'].shape[0]
    if(args.cv_number>len(G_list)):
        vals=len(G_list)
    else:
        vals=args.cv_number
    preds=[]
    labels=[]
    tracked_dicts = []
    for i in range(vals):
        train_dataset, val_dataset, positive_graphs_torch, negative_graphs_torch = cross_val.prepare_data_kfold(G_list, args, i)
        idx_pos, idx_neg = cross_val.count_populations(train_dataset)
        assign_input = num_nodes
        input_dim = num_nodes
        print("CV : ",i)
        model_GTN = GTN(num_edge=num_edge,
                            num_channels=2,
                            num_layers=args.num_layers,
                            norm=True)
        model_DIFFPOOL = encoders.SoftPoolingGcnEncoder(
                    num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input).cpu()
        
        model_LearnableDistance = LearnableDistance(num_nodes = num_nodes)
        tracked_Dict = train(args, train_dataset, val_dataset, model_GTN, model_DIFFPOOL, positive_graphs_torch, negative_graphs_torch)
        #labels.extend(label)
        #preds.extend(pred)
        tracked_dicts.append(tracked_Dict)
    return tracked_dicts
def main():
    args = arg_parse()
    print("Main : ",args)
    tracked_dicts = benchmark_task(args)
    print("finished")
    with open('tracked_dicts_' + args.dataset +'_'+ args.fusion_method, 'wb') as f:
        pickle.dump(tracked_dicts, f)
    '''cm1 = confusion_matrix(labels, preds)
    print('Confusion Matrix : \n', cm1)

    total1=sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Sensitivity : ', sensitivity1 )

    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Specificity : ', specificity1)'''
    # plot loss evolution across epochs
    '''x_epochs = [i for i in range(args.num_epochs)]
    plt1, = plt.plot(x_epochs, Epoch_losses, label='Total loss')
    plt2, = plt.plot(x_epochs, GTN_losses, label='MGI loss')
    plt3, = plt.plot(x_epochs, Epoch_losses, label='DIFFPOOL loss')
    #plt4, = plt.plot(x_epochs, Dissimilarity_losses, label='Dissimilarity loss')
    plt.legend(handles=[plt1,plt2,plt3])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()'''
    
if __name__ == '__main__':
    main()
    
    