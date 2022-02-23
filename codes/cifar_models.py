import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler



NUM_CLASSES = 10


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
    
class Net(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Net, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Sequential(
    
            nn.Linear(128 * 4 * 4, 256),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = x*x
        x = self.pool(x)
        x = self.conv2(x)
        x = x*x
        x = self.pool(x)
        x = self.conv3(x)
        x = x*x
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifier(x)
        x = 1+x+0.5*x**2                                   #Taylor approx of softmax. 안쓸거면 여기서부터는 뺀다
        x /= torch.sum(x,axis=1).view(-1,1)

        return x

class Net_logit(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Net_logit, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Sequential(
    
            nn.Linear(128 * 4 * 4, 256),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = x*x
        x = self.pool(x)
        x = self.conv2(x)
        x = x*x
        x = self.pool(x)
        x = self.conv3(x)
        x = x*x
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifier(x)
#         x = 1+x+0.5*x**2                                   #Taylor approx of softmax. 안쓸거면 여기서부터는 뺀다
#         x /= torch.sum(x,axis=1).view(-1,1)

        return x   

#attack 모델. maxpool relu 다 사용함
class AttackNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AttackNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Sequential(
    
            nn.Linear(128 * 4 * 4, 256),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifier(x)
        return x