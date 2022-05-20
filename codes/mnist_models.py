import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler

from utils import apply_taylor_softmax

NUM_CLASSES = 10

class NetMNIST(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(NetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(5, 50, kernel_size=5, stride=2, padding=0)
        self.classifier = nn.Linear(50*4*4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x*x
        x = self.conv2(x)
        x = x*x
        x = x.view(-1, 50 * 4 * 4)
        x = self.classifier(x)
        x = 1+x+0.5*x**2                                   #Taylor approx of softmax. 안쓸거면 여기서부터는 뺀다
        x /= torch.sum(x,axis=1).view(-1,1)
        return x
    
class AttackNetMNIST(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AttackNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(5, 50, kernel_size=5, stride=2, padding=0)
        self.classifier = nn.Linear(50*4*4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(-1, 50 * 4 * 4)
        x = self.classifier(x)
        return x

class small_NetMNIST(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(small_NetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0)
#        self.conv2 = nn.Conv2d(5, 50, kernel_size=5, stride=2, padding=0)
        self.classifier = nn.Linear(5*12*12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x*x
 #       x = self.conv2(x)
 #       x = x*x
        x = x.view(-1, 5 * 12 * 12)
        x = self.classifier(x)
        x = 1+x+0.5*x**2                                   #Taylor approx of softmax. 안쓸거면 여기서부터는 뺀다
        x /= torch.sum(x,axis=1).view(-1,1)
        return x
class small_AttackNetMNIST(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(small_AttackNetMNIST, self).__init__() #28 28
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0) #5 12 12
#        self.conv2 = nn.Conv2d(5, 50, kernel_size=5, stride=2, padding=0) 50 4 4 
        self.classifier = nn.Linear(5*12*12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
 #       x = self.conv2(x)
 #       x = nn.ReLU()(x)
        x = x.view(-1, 5 * 12 * 12)
        x = self.classifier(x)
        return x