import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Subset

import copy

import os

from torch.utils.data import TensorDataset, DataLoader

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

mnist_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

imagenet_transform=transforms.Compose(
                    [transforms.Resize(32),
                     transforms.ToTensor(),
                     transforms.RandomHorizontalFlip(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def make_st_loader(model, train_loader, device, num_data = None):
    model.to(device)
    model.eval()
    outs = []
    with torch.no_grad():
        for x,_ in train_loader:
            outs.append(model(x.to(device)).detach().cpu())
            del x
    outs = torch.cat(outs,dim=0)
    st_loader = DataLoader(TensorDataset(outs),shuffle=True,batch_size=train_loader.batch_size)
    model.cpu()
    return st_loader   

def make_steal_loader(model, train_loader, device, num_data = None, soft=True):
    model.to(device)
    model.eval()
    outs = []
    xs = []
    with torch.no_grad():
        for x,_ in train_loader:
            if soft:
                outs.append(model(x.to(device)).detach().cpu())
            else:
                outs.append(model(x.to(device)).max(dim=1)[1].detach().cpu())
            xs.append(x.detach().cpu())
            del x
    xs = torch.cat(xs, dim=0)
    outs = torch.cat(outs,dim=0)
    st_loader = DataLoader(TensorDataset(xs,outs),shuffle=True,batch_size=train_loader.batch_size)
    model.cpu()
    return st_loader 
    

def train_valid_split(dataloader, total_data=20000, ratio = 0.5, batch_size = 128,
                        seed=None,datatype ="cifar"):
    torch.manual_seed(seed)
    dset = copy.deepcopy(dataloader.dataset)
    
    if datatype =="cifar":
        dset.transform = transform_test
    else:
        dset.transform = mnist_transform

    if total_data > len(dset.targets):
        total_data = len(dset.targets)
    print("total data:", total_data)
    dset.data = dset.data[:total_data]
    dset.targets = dset.targets[:total_data]
    num_tr = int(total_data*ratio)
    num_val = total_data - num_tr
    trset, valset = torch.utils.data.random_split(dset, [num_tr, num_val])
    trloader = torch.utils.data.DataLoader(trset, batch_size = batch_size, shuffle = True)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle = False)
    return trloader, valloader
    

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

def load_mnist(data_dir="../data/mnist", batch_size=128, test_batch = None,train_shuffle=True):

    trainset = datasets.MNIST(root=data_dir, train=True,
                                            download=True, transform=mnist_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=train_shuffle)#, num_workers=2)
    testset = datasets.MNIST(root=data_dir, train=False,
                                           download=True, transform=mnist_transform)
    
    if test_batch is None:
        test_batch = len(testset)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False)#, num_workers=2)
    return trainloader, testloader


def load_fmnist(data_dir="../data/fmnist", batch_size=128, test_batch = None,train_shuffle=True):

    trainset = datasets.FashionMNIST(root=data_dir, train=True,
                                            download=True, transform=mnist_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=train_shuffle)#, num_workers=2)
    testset = datasets.FashionMNIST(root=data_dir, train=False,
                                           download=True, transform=mnist_transform)
    
    if test_batch is None:
        test_batch = len(testset)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False)#, num_workers=2)
    return trainloader, testloader

def load_emnist_letters(data_dir="../data/emnist_letters", batch_size=128, test_batch = None,train_shuffle=True):

    trainset = datasets.EMNIST(root=data_dir, split='letters', train=True,
                                            download=True, transform=mnist_transform)
    sub_ind = np.random.choice(len(trainset), 50000, replace=False)
    subset = copy.deepcopy(trainset)
    subset.data = trainset.data[sub_ind]
    subset.targets = trainset.targets[sub_ind]
#     subset = Subset(trainset, sub_ind)
    
    trainloader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                              shuffle=train_shuffle)#, num_workers=2)
    testset = datasets.EMNIST(root=data_dir, split='letters', train=False,
                                           download=True, transform=mnist_transform)
    
    if test_batch is None:
        test_batch = len(testset)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False)#, num_workers=2)
    return trainloader, testloader

def load_cifar10(data_dir="../data/cifar10", batch_size=128, test_batch = None,train_shuffle=True):

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
    trainset = datasets.CIFAR100(root=data_dir, train=True,
                                            download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=train_shuffle)#, num_workers=2)
    testset = datasets.CIFAR100(root=data_dir, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)#, num_workers=2)
    return trainloader, testloader

def load_tinyimagenet(data_dir="../data/tiny-imagenet-200", batch_size=128, train_shuffle=True):
#     transform=transforms.Compose(
#                     [transforms.Resize(32),
#                      transforms.ToTensor(),
#                      transforms.RandomHorizontalFlip(),
#                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.ImageFolder(os.path.join(data_dir, "train"),imagenet_transform)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=train_shuffle, num_workers = 100)
    return data_loader