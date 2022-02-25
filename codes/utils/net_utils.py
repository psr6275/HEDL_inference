import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler

def apply_taylor_softmax(x):
    x = 1+x+0.5*x**2                                   
    x /= torch.sum(x,axis=1).view(-1,1)
    return x

class Net_tsoftmax(nn.Module):
    def __init__(self, model):
        super(Net_tsoftmax, self).__init__()
        self.model = model
    def forward(self,x):
        x = self.model(x)
        x = apply_taylor_softmax(x)
        return x

class CombNet(nn.Module):
    def __init__(self, net_orig, net_fake, tau=0.5):
        super(CombNet, self).__init__()
        self.net_orig = net_orig
        self.net_fake = net_fake
        self.tau = tau
        
    def forward(self, x):
        x1 = self.net_orig(x)
        x2 = self.net_fake(x)
        cond_in = torch.max(x1, dim=1).values>self.tau
        out = (x1*cond_in.view(-1,1)+x2*(~cond_in.view(-1,1)))
        return out
    
class CombNet_soft(nn.Module): ## 준영이가 HE 가능하게 max 로 바꿔주기!
    def __init__(self, net_orig, net_fake, tau=0.5):
        super(CombNet, self).__init__()
        self.net_orig = net_orig
        self.net_fake = net_fake
        self.tau = tau
        
    def forward(self, x):
        x1 = self.net_orig(x)
        x2 = self.net_fake(x)
        cond_in = torch.max(x1, dim=1).values>self.tau
        out = (x1*cond_in.view(-1,1)+x2*(~cond_in.view(-1,1)))