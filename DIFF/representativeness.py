# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:49:21 2020

@author: Mohammed Amine
"""
import numpy as np
import pickle
import math


def DistCalc(A,B):
    d = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            d += math.pow((A[i,j] - B[i,j]),2)
    dist = math.sqrt(d)
    return dist

def DistCalcNorm(A, list1):
    dist1 = 0
    dist = 0
    for i in range(len(list1)):
        for v in range(list1[i].shape[2]):
            mat_i = list1[i][:,:,v]
            mean_i = np.mean(mat_i)
            max_i = np.max(mat_i)
            fd = DistCalc(A, mat_i)
            dist1 = (fd - mean_i)/(max_i-mean_i) #+ 1.5
            dist += dist1
    nv = len(list1)*list1[0].shape[2]
    dist /= nv
    return dist

def SeparateTestClasses(list_A, list_labels):
    test_pos = []
    test_neg = []
    for i in range(len(list_A)):
        mat_i = np.squeeze(list_A[i])
        mat_i = mat_i[:,:,:mat_i.shape[2]-1] 
        if list_labels[i]==1:
            test_pos.append(np.squeeze(mat_i))
        if list_labels[i]==0:
            test_neg.append(np.squeeze(mat_i))
        
    return test_pos, test_neg

args_dataset='LH'

with open('tracked_dict_'+args_dataset,'rb') as f:
    tracked_dict = pickle.load(f)

    
    
dist_pos = []
dist_neg = []    
for cv in range(len(tracked_dict)): 
    CBT_pos = tracked_dict[cv]['positive_template']
    CBT_neg = tracked_dict[cv]['negative_template']
    test_A = tracked_dict[cv]['test_A']
    test_labels = tracked_dict[cv]['test_labels']
    
    test_pos, test_neg = SeparateTestClasses(test_A, test_labels)
    
    dist_pos.append(DistCalcNorm(CBT_pos, test_pos))
    dist_neg.append(DistCalcNorm(CBT_neg, test_neg))

std_pos = np.std(dist_pos)
std_neg = np.std(dist_neg)
avg_pos = np.mean(dist_pos)
avg_neg = np.mean(dist_neg)

print(args_dataset)
print(avg_pos)
print(std_pos)
print(avg_neg)
print(std_neg)

print('s')
        
