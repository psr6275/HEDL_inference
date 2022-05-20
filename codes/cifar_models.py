import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler

from utils import apply_taylor_softmax

NUM_CLASSES = 10

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
    
class Net2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Net2, self).__init__()
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
        x = self.pool(x)
        #성능 많이 낮아지면 +linear
        x = x*x
        #print(x[0][0])
        x = self.conv2(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = self.conv3(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = x.view(-1, 128 * 4 * 4)
        logit = self.classifier(x)
        #print(x[0])
        out = 1+logit+0.5*logit**2                                   #Taylor approx of softmax. 안쓸거면 여기서부터는 뺀다
        out /= torch.sum(out,axis=1).view(-1,1)
        #print(x)
        return out

class Net3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Net3, self).__init__()
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
        x = self.pool(x)
        #성능 많이 낮아지면 +linear
        x = x*x
        #print(x[0][0])
        x = self.conv2(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = self.conv3(x)
        x = self.pool(x)
        x = x*x
        #print(x[0][0])
        x = x.view(-1, 128 * 4 * 4)
        out = self.classifier(x)
#         #print(x[0])
#         out = 1+logit+0.5*logit**2                                   #Taylor approx of softmax. 안쓸거면 여기서부터는 뺀다
#         out /= torch.sum(out,axis=1).view(-1,1)
        #print(x)
        return out    
    
class Net_MIA(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Net_MIA, self).__init__()
        self.num_input = 2*num_classes
        self.lin1 = nn.Linear(self.num_input, 128)
        self.lin2 = nn.Linear(128,64)
        self.lin3 = nn.Linear(64,64)
        self.lin4 = nn.Linear(64,1)
    def forward(self,x):
        x = x.view(-1,self.num_input)
        x = F.relu(self.lin1(x))
        x = nn.Dropout(0.3)(x)
        x = F.relu(self.lin2(x))
        x = nn.Dropout(0.2)(x)
        x = F.relu(self.lin3(x))
        x = torch.sigmoid(self.lin4(x)).flatten()
        return x
    
class Net_Adapt(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Net_Adapt, self).__init__()
        self.num_input = num_classes
        self.lin1 = nn.Linear(self.num_input, 128)
        self.lin2 = nn.Linear(128,64)
        self.lin3 = nn.Linear(64,64)
        self.lin4 = nn.Linear(64,1)
    def forward(self,x):
        x = x.view(-1,self.num_input)
        x = F.relu(self.lin1(x))
        x = nn.Dropout(0.3)(x)
        x = F.relu(self.lin2(x))
        x = nn.Dropout(0.2)(x)
        x = F.relu(self.lin3(x))
        x = torch.sigmoid(self.lin4(x)).flatten()
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
     

class LeNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations
    """

    # network architecture:
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
class small_Net2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_conv = 32, hidden_fc = 128, kernel_size = 3):
        super(small_Net2, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, hidden_conv, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.classifier = nn.Sequential(
    
            nn.Linear(hidden_conv * 16 * 16, hidden_fc),
            nn.Linear(hidden_fc, 10),
        )
        self.hidden_conv = hidden_conv
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x*x
        x = x.view(-1, self.hidden_conv * 16 * 16)
        logit = self.classifier(x)
        #print(x[0])
        out = 1+logit+0.5*logit**2                                   #Taylor approx of softmax. 안쓸거면 여기서부터는 뺀다
        out /= torch.sum(out,axis=1).view(-1,1)
        #print(x)
        return out

class small_Net3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_conv = 32, hidden_fc = 128, kernel_size = 3):
        super(small_Net3, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, hidden_conv, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.classifier = nn.Sequential(
    
            nn.Linear(hidden_conv * 16 * 16, hidden_fc),
            nn.Linear(hidden_fc, 10),
        )
        self.hidden_conv = hidden_conv
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x*x
        x = x.view(-1, self.hidden_conv * 16 * 16)
        out = self.classifier(x)
        #print(x[0])
#         out = 1+logit+0.5*logit**2                                   #Taylor approx of softmax. 안쓸거면 여기서부터는 뺀다
#         out /= torch.sum(out,axis=1).view(-1,1)
        #print(x)
        return out

class small_fakenet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_conv = 32, hidden_fc = 128, kernel_size = 3):
        super(small_fakenet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(3, hidden_conv, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_conv * 16 * 16, hidden_fc),
            nn.Linear(hidden_fc, 10),
        )
        self.hidden_conv = hidden_conv
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(-1, self.hidden_conv * 16 * 16)
        x = self.classifier(x)
        return x
