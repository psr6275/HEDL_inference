import torch
import torch.nn as nn
from .train_utils import test_model
from .eval_utils import AverageVarMeter, accuracy, correspondence_score

from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F

import numpy as np
from livelossplot import PlotLosses
import os

from .net_utils import CombNet, CombNet_soft
from .train_utils import test_binary_model

from torch.utils.data import TensorDataset, DataLoader
def make_adaptive_loader(model_prob, victim_net, dataloader, tau, device, batch_size=128, shuffle=True):

    model_prob.to(device).eval()
    victim_net.to(device).eval()
    labels = []
    ori_out = []
    with torch.no_grad():
        for x,y in dataloader:
            with torch.no_grad():
                out = victim_net(x.to(device)).detach().cpu()
                ori_out.append(out)
            pred = torch.max(model_prob(x.to(device)).detach().cpu(),axis=1)
#             print(pred)
            labels.append(torch.tensor(pred[0]>tau).float())
            del x,y,pred, out
    data = torch.cat(ori_out,dim=0)
    labels = torch.cat(labels,dim=0)
    dataloader_ = DataLoader(TensorDataset(data, labels),batch_size=batch_size, shuffle=shuffle)
    model_prob.cpu()
    victim_net.cpu()
    return dataloader_



def train_adapt_model(adpt_att_model, adpt_loader, criterion, optimizer, epochs, device, adpt_testloader = None, save_dir = "../results", save_model = "cifar_adapt_model.pth"):
    adpt_att_model.to(device)
    
    logs_clf = {}
    best_acc = 0.0
    liveloss_tr = PlotLosses()
    
    for epoch in range(epochs):
        adpt_att_model.train()
        
        for x,y in adpt_loader:

            adpt_att_model.zero_grad()

            out = adpt_att_model(x.to(device))
            loss = criterion(out.flatten().float(), y.to(device).float())
            
            loss.backward()
            optimizer.step()
            
            del x,out,  y, loss
            torch.cuda.empty_cache()
        logs_clf['loss'], logs_clf['acc']= test_binary_model(adpt_att_model, adpt_loader, criterion, device, 100.0, save_dir, save_model)
        if adpt_testloader is not None:
            logs_clf['val_loss'], logs_clf['val_acc']= test_binary_model(adpt_att_model, adpt_testloader, criterion, device, 0.0, save_dir, save_model)
        liveloss_tr.update(logs_clf)
        liveloss_tr.send()
    adpt_att_model.cpu()
    return adpt_att_model, logs_clf            

def select_data(trainset, nb_stolen,batch_size=128, select_shuffle = False):  #attack??? ????????? ??? ????????? ?????? ??????
    x = trainset.data
    nb_stolen = np.minimum(nb_stolen, x.shape[0])
    rnd_index = np.random.choice(x.shape[0], nb_stolen, replace=False)
    sampler = SubsetRandomSampler(rnd_index)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=select_shuffle, sampler=sampler)
    return trainloader

def query_label(x, victim_clf, victim_clf_fake, tau, use_probability=False):   #tau ????????? ?????? net ?????? fakenet ?????????
    victim_clf.to(device).eval()
    victim_clf_fake.to(device).eval()
    labels_in = victim_clf(x)
    labels_out = victim_clf_fake(x)
    cond_in = torch.max(labels_in, dim=1).values>tau
    labels = (labels_in*cond_in.view(-1,1)+labels_out*(~cond_in.view(-1,1)))
    
    if not use_probability:
        labels = torch.argmax(labels, axis=1)
        #labels = to_categorical(labels, nb_classes)
    
    victim_clf.cpu()
    victim_clf_fake.cpu()
    
    return labels


def train_stmodel_comb_hard_label(steal_loader, thieved_clf, criterion, optimizer, victim_comb, epochs, device, 
                      test_loader=None, save_dir = "../results", save_model="cifar_stmodel.pth"):
    """
        attack with hard label
    """
    
    thieved_clf.to(device)
    liveloss_tr = PlotLosses()
    logs_clf = {}
    best_acc = 0.0
    
#     save_model = str(victim_comb.tau)+"_"+save_model
    for epoch in range(epochs):
        losses = AverageVarMeter()
        thieved_clf.train()
        for x,y in steal_loader:
            x = x.to(device)
            
            victim_comb.to(device).eval()
            fake_out = victim_comb(x)
            victim_comb.cpu()
            
            thieved_clf.zero_grad()
            out = thieved_clf(x)
            
            
            _,fake_label = torch.max(fake_out,dim=1)
                
            loss = criterion(out,fake_label)
            
            loss.backward()
            optimizer.step()
            
            losses.update(loss,x.size(0))
            del out,x,y,loss, fake_label
            torch.cuda.empty_cache()
        logs_clf['loss'] = losses.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss'],logs_clf['val_acc'] = test_model(thieved_clf, test_loader, criterion, device, 0.0, save_dir, save_model)
            if best_acc<logs_clf['val_acc']:
                best_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()  
    return thieved_clf, logs_clf     


def train_stmodel_comb_soft_label(steal_loader, thieved_clf, criterion, optimizer, victim_comb, epochs, device, 
                      test_loader=None, save_dir = "../results", save_model="cifar_stmodel.pth"):
    """
        attack with hard label
    """
    
    thieved_clf.to(device)
    liveloss_tr = PlotLosses()
    logs_clf = {}
    best_acc = 0.0
    
#     save_model = str(victim_comb.tau)+"_"+save_model
    for epoch in range(epochs):
        losses = AverageVarMeter()
        thieved_clf.train()
        for x,y in steal_loader:
            x = x.to(device)
            
            victim_comb.to(device).eval()
            fake_out = victim_comb(x)
            victim_comb.cpu()
            
            thieved_clf.zero_grad()
            out = thieved_clf(x)
                
            loss = criterion(out,fake_out)
            
            loss.backward()
            optimizer.step()
            
            losses.update(loss,x.size(0))
            del out,x,y,loss, fake_label
            torch.cuda.empty_cache()
        logs_clf['loss'] = losses.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss'],logs_clf['val_acc'] = test_model(thieved_clf, test_loader, criterion, device, 0.0, save_dir, save_model)
            if best_acc<logs_clf['val_acc']:
                best_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()  
    return thieved_clf, logs_clf     
    


    
def train_stmodel(steal_loader, thieved_clf, criterion, use_probability, optimizer, victim_clf, victim_clf_fake, tau, nb_stolen, batch_size, epochs, device, testloader=None, save_dir = "../results", save_model="cifar_stmodel.pth"):
    
    thieved_clf.to(device)
    liveloss_tr = PlotLosses()
    logs_clf = {}
    best_acc = 0.0
    
    save_model = str(tau)+"_"+save_model
    for epoch in range(epochs):
        losses = AverageVarMeter()
        thieved_clf.train()
        for x,y in steal_loader:
            x = x.to(device)
            
            thieved_clf.zero_grad()
            
            out = thieved_clf(x)
            fake_label = query_label(x, victim_clf, victim_clf_fake, tau, use_probability)

            loss = criterion(out,fake_label)
            
            loss.backward()
            optimizer.step()
            
            losses.update(loss,x.size(0))
            del out,x,y,loss, fake_label
            torch.cuda.empty_cache()
        logs_clf['loss'] = losses.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss'],logs_clf['val_acc'] = test_model(thieved_clf, test_loader, criterion, device, best_acc, save_dir, save_model)
            if best_acc<logs_clf['val_acc']:
                best_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()  
    return thieved_clf, logs_clf
    
    
def test_model_from_taus(model, model_fake, tau_list, test_loader, criterion, device, soft = False):
    lss = []
    acs = []
    for tau in tau_list:
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        if soft:
            model_comb = CombNet_soft(model, model_fake,tau).to(device).eval()
        else:
            model_comb = CombNet(model, model_fake,tau).to(device).eval()
        for batch_idx, (x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            p_y = model_comb(x)
            loss = criterion(p_y,y)

            acc = accuracy(p_y.detach().cpu(), y.detach().cpu())

            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
    #         print(acc)
            del x, y, p_y, acc, loss
            torch.cuda.empty_cache()
        lss.append(losses.avg.detach().cpu())
        acs.append(accs.avg.detach().cpu())
        print("Accuracy/Loss for tau {:.1f} : {:.2f}/{:.4f}".format(tau,acs[-1],lss[-1]))
        del model_comb
        torch.cuda.empty_cache()
    return lss, acs

def test_corr_model(model1, model2, test_loader, criterion, device):
    
    model1.to(device).eval()
    model2.to(device).eval()
    
    losses1 = AverageVarMeter()
    losses2 = AverageVarMeter()
    accs1 = AverageVarMeter()
    accs2 = AverageVarMeter()
    corrs = AverageVarMeter()
    for batch_idx, (x,y) in enumerate(test_loader):
        x = x.to(device)

        p_y1 = model1(x).detach().cpu()
        p_y2 = model2(x).detach().cpu()

        loss1 = criterion(p_y1,y)
        loss2 = criterion(p_y2,y)

        acc1 = accuracy(p_y1, y)
        acc2 = accuracy(p_y2, y)
        
        corr = correspondence_score(p_y1,p_y2)

        losses1.update(loss1,x.size(0))
        losses2.update(loss2,x.size(0))
        accs1.update(acc1[0],x.size(0))
        accs2.update(acc2[0],x.size(0))
        corrs.update(corr,x.size(0))
#         print(acc)
        del x, y, p_y1, p_y2, acc1,acc2, loss1, loss2, corr
        torch.cuda.empty_cache()
    loss1 = losses1.avg.detach().cpu()
    loss2 = losses2.avg.detach().cpu()
    acc1 = accs1.avg.detach().cpu()
    acc2 = accs2.avg.detach().cpu()
    corr = corrs.avg.detach().cpu()
    
    print("Accuracy/Loss 1: {:.2f}/{:.4f}".format(acc1,loss1))
    print("Accuracy/Loss 2: {:.2f}/{:.4f}".format(acc2,loss2))
    print("Correspondence: ", corr)
    
    del losses1, losses2, accs1, accs2, corrs

    return loss1, loss2, acc1, acc2, corr