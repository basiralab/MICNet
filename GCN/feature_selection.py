# -*- coding: utf-8 -*-
"""
Created on Fri May 22 02:05:44 2020

@author: Mohamed
"""

import numpy as np
import pickle
import pandas as pd
import math
from sklearn import svm
from sklearn.metrics import confusion_matrix

with open('tracked_dict_LH','rb') as f:
    tracked_dicts = pickle.load(f)

def DistCalc(A, list1):
    dist = 0
    for i in range(len(list1)):
        dist += np.linalg.norm(A-list1[i])
    return dist

def SeparatePop(train_H, train_labels, test_H, test_labels):
    pos_pop = []
    neg_pop = []
    for i in range(len(train_H)):
        if train_labels[i]==0:
            neg_pop.append(train_H[i])
        if train_labels[i]==1:
            pos_pop.append(train_H[i])
    for i in range(len(test_H)):
        if test_labels[i]==0:
            neg_pop.append(test_H[i])
        if test_labels[i]==1:
            pos_pop.append(test_H[i])
    return pos_pop, neg_pop
            
def get_cv_train(tracked_Dict,i):
    train_labels = tracked_Dict[i]['train_labels'] 
    train_H = tracked_Dict[i]['train_H'] 
    train_labels2 =  [train_labels[i][0] for i in range(len(train_labels))]
    return train_H, train_labels2

def get_cv_test(tracked_Dict,i):
    test_labels = tracked_Dict[i]['test_labels']
    test_H = tracked_Dict[i]['test_H'] 
    test_labels2 =  test_labels.tolist()
    return test_H, test_labels2

def get_cv_templates(tracked_Dict,i):
    positive_template = tracked_Dict[i]['positive_template'] 
    negative_template = tracked_Dict[i]['negative_template'] 
    return positive_template, negative_template

def get_upper_triangle(A):
    B = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if(i>j):
                B[i,j]=0
            else:
                B[i,j]=A[i,j]
    return B

def max_n(A,n):
    L = np.zeros((A.shape[0], A.shape[1]))
    C = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            L[i,j] = i
            C[i,j] = j
    flat_A = np.squeeze(np.asarray(A.flatten())).tolist()
    flat_L = np.squeeze(np.asarray(L.flatten())).tolist()
    flat_C = np.squeeze(np.asarray(C.flatten())).tolist()

    zipped_lists = zip(flat_A, flat_L, flat_C) #Pair list2 and list1 elements.
    sorted_zipped_lists = sorted(zipped_lists, reverse=True) #Sort by first element of each pair.
    indexes = []
    for i in range(n):
        indexes_local = [int(sorted_zipped_lists[i][1]), int(sorted_zipped_lists[i][2])]
        indexes.append(indexes_local)
    return indexes
            
inter_pos = []
inter_neg = []
intra_pos_pop = []
intra_neg_pop = []
for k in range(10,500,10):
    preds = []
    targets = []
    inter_pos_cv = []
    inter_neg_cv = []
    intra_pos_pop_cv = []
    intra_neg_pop_cv = []
    for cv in range(4):
        train_H, train_labels = get_cv_train(tracked_dicts,cv)
        test_H, test_labels = get_cv_test(tracked_dicts,cv)
        positive_template, negative_template = get_cv_templates(tracked_dicts,cv)
        
        difference_template = np.abs(positive_template-negative_template)
        difference_template_ut = get_upper_triangle(difference_template)
        
        pos_pop, neg_pop = SeparatePop(train_H, train_labels, test_H, test_labels)
        
        inter_pos_cv.append(DistCalc(positive_template, pos_pop))
        inter_neg_cv.append(DistCalc(negative_template, neg_pop))
        intra_pos_pop_cv.append(DistCalc(positive_template, neg_pop))
        intra_neg_pop_cv.append(DistCalc(negative_template, pos_pop))
        
        max_indexes_ut = max_n(difference_template_ut, k)
        train_features = []
        test_features = []
        for i in range(len(train_H)):
            train_features_local = []
            for j in range(len(max_indexes_ut)):
                train_features_local.append(train_H[i][max_indexes_ut[j][0],max_indexes_ut[j][1]])
            train_features.append(train_features_local)

        for i in range(len(test_H)):
            test_features_local = []
            for j in range(len(max_indexes_ut)):
                test_features_local.append(test_H[i][max_indexes_ut[j][0],max_indexes_ut[j][1]])
            test_features.append(test_features_local)
    
        clf = svm.SVC(kernel='linear')
        clf.fit(train_features, train_labels)
        test_preds = clf.predict(test_features)
        preds.extend(test_preds)
        targets.extend(test_labels)
    
    inter_pos.append(sum(inter_pos_cv) / len(inter_pos_cv))
    inter_neg.append(sum(inter_neg_cv) / len(inter_neg_cv))
    intra_pos_pop.append(sum(intra_pos_pop_cv) / len(intra_pos_pop_cv))
    intra_neg_pop.append(sum(intra_neg_pop_cv) / len(intra_neg_pop_cv))
    cm1 = confusion_matrix(targets, preds)
    #print('Confusion Matrix : \n', cm1)
    total1=sum(sum(cm1))
    
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Accuracy : ', accuracy1)
    
    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Sensitivity : ', sensitivity1 )
    
    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Specificity : ', specificity1)
    
print("s")
