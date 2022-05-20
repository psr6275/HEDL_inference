import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
import copy

def inv_apprx(x, t, m):
    a = 2-2/m*x
    b = 1-2/m*x
    for i in range(t):
        b = b*b
        a = a*(1+b)
    return 2/m*a

def comp_max_tau(output, tau, t1, t2):
    device = output.device
    res = copy.copy(output)
    #print(res.shape)
    #print((tau*torch.ones(len(res)).view(-1,1)).shape)
    res = torch.cat((res, tau*torch.ones(len(res)).view(-1,1).to(device)), 1)
    for i in range(t1):
        res = res*res
        sum_res = res.sum(1)
        if i==0:
            inv = inv_apprx(sum_res, t2, 2+tau*tau)
        else:
            inv = inv_apprx(sum_res, t2, 2)
        res *= inv.reshape(-1,1)
    return res[:,-1]



def apply_taylor_softmax(x,emph=1.0):
    x = (1+x+0.5*x**2)**emph                                   
    x /= torch.sum(x,axis=1).view(-1,1)
    return x

class Net_tsoftmax(nn.Module):
    def __init__(self, model,temp = 1000.0):
        super(Net_tsoftmax, self).__init__()
        self.model = model
        self.temp = temp
    def forward(self,x):
        x = self.model(x)
        x = apply_taylor_softmax(x/self.temp)
        return x
    
class Net_softmax(nn.Module):
    def __init__(self, model):
        super(Net_softmax, self).__init__()
        self.model = model
    def forward(self,x):
        x = self.model(x)
        x = F.softmax(x,dim=1)
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
    
class CombNet_logit(nn.Module): ## 준영이가 HE 가능하게 max 로 바꿔주기!
    def __init__(self, net_orig, net_fake, tau=0.5):
        super(CombNet_logit, self).__init__()
        self.net_orig = Net_softmax(net_orig)
        self.net_fake = Net_softmax(net_fake)
        self.tau = tau
        
    def forward(self, x):
        x1 = self.net_orig(x)
        x2 = self.net_fake(x)
        cond_in = torch.max(x1, dim=1).values>self.tau
        out = (x1*cond_in.view(-1,1)+x2*(~cond_in.view(-1,1)))
        return out

class CombNetHE(nn.Module):
    def __init__(self, net_orig, net_fake, tau=0.5, t1=3, t2=3):
        super(CombNetHE, self).__init__()
        self.net_orig = net_orig
        self.net_fake = net_fake
        self.tau = tau
        self.t1 = t1
        self.t2 = t2
    def forward(self, x):
        x1 = self.net_orig(x)
        x2 = self.net_fake(x)
        #cond_in = torch.max(x1, dim=1).values>self.tau
        cond_in = comp_max_tau(x1, self.tau, self.t1, self.t2)
        out = (x1*(1-cond_in.view(-1,1))+x2*cond_in.view(-1,1))
        return out
    
class CombNetHE_invapp(nn.Module):
    def __init__(self, net_orig, net_fake, tau=0.5, thr = 30, t1=3, t2=3):
        super(CombNetHE_invapp, self).__init__()
        self.net_orig = net_orig
        self.net_fake = net_fake
        self.tau = tau
        self.thr = thr
        self.t1 = t1
        self.t2 = t2
    def forward(self, x):
        x1, _ = self.net_orig(x)
        x1 = 1+x1+0.5*x1**2
        sum_x1 = x1.sum(1)
        invsum_x1 = inv_apprx(sum_x1, self.t2, self.thr)
        x1 *= invsum_x1.reshape(-1,1)
        x2 = self.net_fake(x)
        #cond_in = torch.max(x1, dim=1).values>self.tau
        cond_in = comp_max_tau(x1, self.tau, self.t1, self.t2)
        out = (x1*(1-cond_in.view(-1,1))+x2*cond_in.view(-1,1))
        return out
    
class CombNet_soft(nn.Module): ## 준영이가 HE 가능하게 max 로 바꿔주기!
    def __init__(self, net_orig, net_fake, tau=0.5,nu=10):
        super(CombNet_soft, self).__init__()
        self.net_orig = net_orig
        self.net_fake = net_fake
        self.tau = tau
        self.nu = nu
        
    def forward(self, x):
        x1 = self.net_orig(x)
        x2 = self.net_fake(x)
        cond_in = torch.sigmoid(self.nu*(self.tau-torch.max(x1, dim=1).values))
        out = (x1*(1-cond_in.view(-1,1))+x2*(cond_in.view(-1,1)))
        return out

