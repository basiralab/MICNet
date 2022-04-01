# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:58:29 2020

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
from models_GCN_average import GCN

import time

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import mlab
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate(dataset, model_GCN, args, name='Test', max_num_examples=None):
    
    model_GCN.eval()
    tracked_Dict = {}
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        CBT_subject = torch.mean(adj,3)
        CBT_subject = torch.squeeze(CBT_subject)
        
        features = np.identity(CBT_subject.shape[0])
        
        features = Variable(torch.from_numpy(features).float(), requires_grad=False).cpu()
        
        if(args.threshold == 'mean'):
            threshold = torch.mean(CBT_subject)
            CBT_subject = torch.where(CBT_subject > threshold, torch.tensor([1.0]), torch.tensor([0.0]))
                
        if(args.threshold == 'median'):
            threshold = torch.median(CBT_subject)
            CBT_subject = torch.where(CBT_subject > threshold, torch.tensor([1.0]), torch.tensor([0.0]))
        
        ypred = model_GCN(features, CBT_subject)

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

def train(args, train_dataset, val_dataset, model_GCN):
    Epoch_losses = []    
    GTN_losses = []
    params = list(model_GCN.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    tracked_Dicts = []
    for epoch in range(args.num_epochs):
        print("Epoch ",epoch)
        
        model_GCN.train()
        total_time = 0
        avg_loss = 0.0
        
        preds = []
        labels = []
        for batch_idx, data in enumerate(train_dataset):
            begin_time = time.time()
            
            
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            adj_id = Variable(data['id'].int()).to(device)
            
            
            
            CBT_subject = torch.mean(adj,3)
            CBT_subject = torch.squeeze(CBT_subject)
            # added
            features = np.identity(CBT_subject.shape[0])
            features = Variable(torch.from_numpy(features).float(), requires_grad=False).cpu()
            
            if(args.threshold == 'mean'):
                threshold = torch.mean(CBT_subject)
                CBT_subject = torch.where(CBT_subject > threshold, torch.tensor([1.0]), torch.tensor([0.0]))
                
            if(args.threshold == 'median'):
                threshold = torch.median(CBT_subject)
                CBT_subject = torch.where(CBT_subject > threshold, torch.tensor([1.0]), torch.tensor([0.0]))
            
            
            
            ypred = model_GCN(features, CBT_subject)
            
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            
            
            loss = model_GCN.loss(ypred, label)
            
            
            model_GCN.zero_grad()
            
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
        tracked_Dict = evaluate(val_dataset, model_GCN, args, name='Test', max_num_examples=100)
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
    if args.NormalizeInputGraphs==True:
        for subject in range(len(adjacencies)):
            for view in range(adjacencies[0].shape[2]):
                adjacencies[subject][:,:,view] = minmax_sc(adjacencies[subject][:,:,view])
    #Add Id view in every subject
    num_nodes = adjacencies[0].shape[0]
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
    parser.add_argument('--dataset', type=str, default='LH_GSP',
                        help='Dataset')
    parser.add_argument('--fusion_method', type=str, default='average',
                        help='Fusion Method')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Training Epochs')
    parser.add_argument('--num_layers', type=int, default=1, #2
                        help='number of layer')
    parser.add_argument('--batch-size', type=int, default=1, # only works with batchsize=1
                        help='Batch size.')
    parser.add_argument('--cv_number', type=int, default=5,
                        help='number of validation folds.')
    parser.add_argument('--lambda_GTN', type=float, default=1,
                        help='multiplication term for loss.')
    parser.add_argument('--lambda_Fusion', type=float, default=0.1,
                        help='multiplication term for loss.')
    parser.add_argument('--IdentityMatrix', default=True, action='store_true',
                        help='Add Identity matrix View in adjacency')
    ##################
    parser.add_argument('--threshold', dest='threshold', default='mean',
            help='threshold the MGI output into binary matrix. Possible values: no_threshold, median, mean')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=2,
                        help='Number of label classes')
    parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
    parser.add_argument('--NormalizeInputGraphs', default=False, action='store_true',
                        help='Normalize Input adjacency matrices of graphs')
    
    
    
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
        
        model_GCN = GCN(nfeat = num_nodes,
                        nhid = args.hidden,
                        nclass = args.num_classes,
                        dropout = args.dropout)
        
        
        tracked_Dict = train(args, train_dataset, val_dataset, model_GCN)
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
    
    