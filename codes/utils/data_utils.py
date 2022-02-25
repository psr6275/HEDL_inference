import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler

import os

from torch.utils.data import TensorDataset, DataLoader

def make_st_loader(model, train_loader, device):
    model.to(device)
    model.eval()
    outs = []
    for x,_ in train_loader:
        outs.append(model(x.to(device)).detach().cpu())
    outs = torch.cat(outs,dim=0)
    st_loader = DataLoader(TensorDataset(outs),shuffle=True,batch_size=train_loader.batch_size)
    model.cpu()
    return st_loader    

def make_randomlabel(dataloader, num_class = 10, batch_size=128, train_shuffle=True):
    randlabel = torch.tensor([])
    trainset = dataloader.dataset
    for i in range(len(trainset.targets)):
        ran = torch.randint(high=num_class, size=(1,))
        while ran[0]==trainset.targets[i]:
            ran = torch.randint(high=10, size=(1,))
        randlabel = torch.cat((randlabel, ran))
    trainloader_all = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.targets))
    train_all = iter(trainloader_all).next()
    randset = torch.utils.data.TensorDataset(torch.tensor(train_all[0]),randlabel.type(torch.LongTensor))
    randloader = torch.utils.data.DataLoader(randset,batch_size=batch_size, shuffle=train_shuffle)
    del trainloader_all, train_all
    return randloader

def load_cifar10(data_dir="../data/cifar10", batch_size=128, test_batch = None,train_shuffle=True):
    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


    trainset = datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=train_shuffle)#, num_workers=2)
    testset = datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform_test)
    
    if test_batch is None:
        test_batch = len(testset)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False)#, num_workers=2)
    return trainloader, testloader

def load_cifar100(data_dir="../data/cifar100", batch_size=128, train_shuffle=False):
    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


    trainset = datasets.CIFAR100(root=data_dir, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=train_shuffle)#, num_workers=2)
    testset = datasets.CIFAR100(root=data_dir, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)#, num_workers=2)
    return trainloader, testloader

def load_tinyimagenet(data_dir="../data/tiny-imagenet-200", batch_size=128, train_shuffle=True):
    transform=transforms.Compose(
                    [transforms.Resize(32),
                     transforms.ToTensor(),
                     transforms.RandomHorizontalFlip(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.ImageFolder(os.path.join(data_dir, "train"),transform)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=train_shuffle, num_workers = 100)
    return data_loader