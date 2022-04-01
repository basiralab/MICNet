# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:37:09 2020

@author: Mohammed Amine
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class learnable_mask(nn.Module):
    def __init__(self, num_nodes):
        super(learnable_mask, self).__init__()
        self.num_nodes = num_nodes
        self.weights_mask = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device))
        
    def forward(self, CBT_subject):
        weights_mask = self.weights_mask
        norm_weights_mask = torch.div(weights_mask-torch.min(weights_mask),torch.max(weights_mask)-torch.min(weights_mask))
        output = torch.mul(CBT_subject, norm_weights_mask)
        output = torch.div(output-torch.mean(output) , torch.std(output))
        #output = torch.div(output-torch.min(output),torch.max(output)-torch.min(output))
        output = F.sigmoid(output)
        threshold = torch.tensor([0.5])
        output = torch.where(output > threshold, torch.tensor([1.0]), torch.tensor([0.0]))
        return output