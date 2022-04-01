# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:11:59 2020

@author: Mohammed Amine
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

A = torch.tensor([[1.0,2.0,5.0],[1.6,1.9,2.4],[1.6,1.1,2.3]])
print(A)
B = torch.div(A-torch.min(A),torch.max(A)-torch.min(A))
C = torch.div(A-torch.mean(A) , torch.std(A))
print(C)
