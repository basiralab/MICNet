# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:04:48 2020

@author: Mohammed Amine
"""


import numpy as np 
import pickle
import torch 


dataset_original = 'LH_GSP'
dataset_threshold = 'LH_GSP_t'


CBT_subject = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
#print(CBT_subject)
threshold = torch.mean(CBT_subject)
CBT_subject = torch.where(CBT_subject > threshold, torch.tensor([1]), torch.tensor([0]))

torch.median()

print(CBT_subject)
print("finished")

