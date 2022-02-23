import torch
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.optim as optim

import numpy as np
from livelossplot import PlotLosses
import os

from .swae_utils import sliced_wasserstein_distance
from .eval_utils import AverageVarMeter, accuracy

def train_swd_fakenet_NLL(clf, train_loader, st_loader, optimizer,device, epochs, 
                          loss_weights=5., test_loader = None,save_dir='../results', save_model="cifar_fake.pth"):

    clf.to(device)
    
    loss_clf = nn.NLLLoss()
    liveloss_tr = PlotLosses()
    logs_clf = {}
    worst_acc = 100.0
    
    for epoch in range(epochs):
        losses = AverageVarMeter()
        losses1 = AverageVarMeter()
        losses2 = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        
        iterloader = iter(st_loader)
        
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            clf.zero_grad()
            
            out = 1-clf(x)
            out = torch.log(out)
            fake_loss = loss_clf(out,y)
#             fake_loss = 0
            try:
                batch = next(iterloader)
            except StopIteration:
                iterloader = iter(st_loader)
                batch = next(iterloader)
            swd_loss = sliced_wasserstein_distance(out,torch.log(batch[0]).to(device),num_projections=50,p=2,device=device)
            loss = fake_loss + swd_loss*loss_weights
            loss.backward()
            optimizer.step()
            acc = accuracy(-out.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            losses1.update(fake_loss.detach().cpu(),x.size(0))
            losses2.update(swd_loss.detach().cpu(),x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss,batch
            torch.cuda.empty_cache()
            
        logs_clf['acc'] = accs.avg.detach().cpu()
        logs_clf['loss'] = losses.avg.detach().cpu()
        logs_clf['loss1'] = losses1.avg.detach().cpu()
        logs_clf['loss2'] = losses2.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss1'],logs_clf['val_acc'] = test_fake_model_NLL(clf, test_loader, loss_clf, device,worst_acc, save_dir, save_model)
            if worst_acc>logs_clf['val_acc']:
                worst_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()
    clf.cpu()
    return clf, logs_clf

def train_swd_fakenet(clf, train_loader, st_loader, optimizer,device, epochs, 
                          loss_weights=[-1.0,5.0],test_loader = None,save_dir='../results', save_model="cifar_fake.pth"):
    clf.to(device)
    loss_clf = nn.CrossEntropyLoss()
    liveloss_tr = PlotLosses()
    logs_clf = {}
    worst_acc = 100.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        losses1 = AverageVarMeter()
        losses2 = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        
        iterloader = iter(st_loader)
        
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            clf.zero_grad()
            out = clf(x)
            fake_loss = loss_clf(out,y)*loss_weights[0]
#             fake_loss = 0
            try:
                batch = next(iterloader)
            except StopIteration:
                iterloader = iter(st_loader)
                batch = next(iterloader)
            swd_loss = sliced_wasserstein_distance(out,batch[0].to(device),num_projections=50,p=2,device=device)
            loss = fake_loss + swd_loss*loss_weights[1]
            loss.backward()
            optimizer.step()
            acc = accuracy(out.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            losses1.update(fake_loss,x.size(0))
            losses2.update(swd_loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss,batch
            torch.cuda.empty_cache()
            
        logs_clf['acc'] = accs.avg.detach().cpu()
        logs_clf['loss'] = losses.avg.detach().cpu()
        logs_clf['loss1'] = losses1.avg.detach().cpu()
        logs_clf['loss2'] = losses2.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss1'],logs_clf['val_acc'] = test_fake_model(clf,test_loader,loss_clf,loss_weights[0],device,worst_acc,save_dir, save_model)
            if worst_acc>logs_clf['val_acc']:
                worst_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send() 
    
    clf.cpu()
    
    return clf, logs_clf

def train_fakenet_NLL(clf, train_laoder, optimizer, device, epochs, test_loader = None, save_dir = "../results",save_model="cifar_fakenet.pth"):
    
    loss_clf = nn.NLLLoss() 
    
    clf.to(device)
    liveloss_tr = PlotLosses()
    logs_clf = {}
    worst_acc = 100.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            clf.zero_grad()
            out = 1-clf(x) 
            #torch에서 logsoftmax 쓰듯이 log를 취해주면 학습이 더 잘됨
            loss = loss_clf(torch.log(out),y)
            loss.backward()
            optimizer.step()
            
            acc = accuracy(1-out.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss
            torch.cuda.empty_cache()
        logs_clf['acc'] = accs.avg.detach().cpu()
        logs_clf['loss'] = losses.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss'],logs_clf['val_acc'] = test_fake_model_NLL(clf,test_loader,loss_clf,device,worst_acc,save_dir,save_model)
            if worst_acc>logs_clf['val_acc']:
                worst_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()  
    
    clf.cpu()
    
    return clf, logs_clf

def test_fake_model_NLL(model, test_loader, criterion, device, worst_acc=100.0, save_dir='../results', save_model = "fake_ckpt.pth"):
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    for batch_idx, (x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        p_y = 1-model(x)
        loss = criterion(torch.log(p_y),y)
        
        acc = accuracy(1-p_y.detach().cpu(), y.detach().cpu())
    
        losses.update(loss,x.size(0))
        accs.update(acc[0],x.size(0))
#         print(acc)
    if accs.avg<worst_acc:
        torch.save(model.state_dict(),os.path.join(save_dir,save_model))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu()


def test_fake_model(model, test_loader, criterion, loss_weight, device, worst_acc=100.0, save_dir='../results', save_model = "fake_ckpt.pth"):
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    for batch_idx, (x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        p_y = model(x)
        loss = loss_weight*criterion(p_y,y)
        
        acc = accuracy(p_y.detach().cpu(), y.detach().cpu())
    
        losses.update(loss,x.size(0))
        accs.update(acc[0],x.size(0))
#         print(acc)
    if accs.avg<worst_acc:
        torch.save(model.state_dict(),os.path.join(save_dir,save_model))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu()

def train_model_multiLoss(clf, train_loader, optimizer, device, 
                          loss_list, loss_weights,epochs, test_loader = None, 
                          save_dir = '../results', save_model="cifar_clf.pth"):
    clf.to(device)
    liveloss_tr = PlotLosses()
    logs_clf = {}
    best_acc = 0.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            clf.zero_grad()
            out = clf(x)
            loss = 0.0
            for i in range(len(loss_list)):                
                loss += loss_list[i](out,y)*loss_weights[i]
            loss.backward()
            optimizer.step()
            acc = accuracy(out.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss
        logs_clf['acc'] = accs.avg.detach().cpu()
        logs_clf['loss'] = losses.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss'],logs_clf['val_acc'] = test_model_multiple(clf, test_loader,loss_list, loss_weights, device, best_acc, save_dir, save_model)
            if best_acc<logs_clf['val_acc']:
                best_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()  
    
    clf.cpu()
    
    return clf, logs_clf

def train_model_with_oe(clf, train_loader, outlier_loader, optimizer, device, 
                          loss_in, loss_out, weight_out, epochs,pred_prob = True, test_loader = None,
                          save_dir = '../results', save_model="cifar_clf.pth"):
    clf.to(device)
    
    liveloss_tr = PlotLosses()
    logs_clf = {}
    best_acc = 0.0
    
    for epoch in range(epochs):
        losses1 = AverageVarMeter()
        losses2 = AverageVarMeter()
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for in_set, out_set in zip(train_loader, outlier_loader):
            x,y = in_set[0].to(device),in_set[1].to(device)
            x_out = out_set[0].to(revice)
            
            clf.zero_grad()
            
            pred_in = clf(x)
            pred_out = clf(x_out)
            
            if pred_prob:
                pred_in = torch.log(pred_in)
            loss1 = loss_in(pred_in,y)
            
            loss2 = weight_out*loss_out(pred_out, torch.ones_like(pred_out)*0.1)
            loss = loss1+loss2
            
            loss.backward()
            optimizer.step()
            
            acc = accuracy(pred_in.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            losses1.update(loss1.detach().cpu(),x.size(0))
            losses2.update(loss2.detach().cpu(),x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, pred_in,x,y,loss, pred_out, x_out
            torch.cuda.empty_cache()
        logs_clf['acc'] = accs.avg.detach().cpu()
        logs_clf['loss'] = losses.avg.detach().cpu()
        logs_clf['loss1'] = losses1.avg.detach().cpu()
        logs_clf['loss2'] = losses2.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss1'],logs_clf['val_acc'] = test_model(clf,test_loader,loss_in,device,best_acc,save_dir,save_model,pred_prob)
            if best_acc<logs_clf['val_acc']:
                best_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()  
    
    clf.cpu()
    
    return clf, logs_clf

def train_model_CE(clf, train_loader, optimizer, device, epochs, test_loader = None, save_dir = "../results",save_model="cifar_clf.pth"):
    
    loss_clf = nn.CrossEntropyLoss()
    clf.to(device)
    liveloss_tr = PlotLosses()
    logs_clf = {}
    best_acc = 0.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            clf.zero_grad()
            out = clf(x)
            loss = loss_clf(out,y)
            loss.backward()
            optimizer.step()
            acc = accuracy(out.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss
            torch.cuda.empty_cache()
        logs_clf['acc'] = accs.avg.detach().cpu()
        logs_clf['loss'] = losses.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss'],logs_clf['val_acc'] = test_model(clf,test_loader,loss_clf,device,best_acc,save_dir,save_model,False)
            if best_acc<logs_clf['val_acc']:
                best_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()  
    
    clf.cpu()
    
    return clf, logs_clf

def train_model_NLL(clf, train_loader, optimizer, device, epochs, test_loader = None, save_dir = "../results",save_model="cifar_clf.pth"):
    
    loss_clf = nn.NLLLoss() 
    
    clf.to(device)
    liveloss_tr = PlotLosses()
    logs_clf = {}
    best_acc = 0.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            clf.zero_grad()
            out = clf(x) 
            #torch에서 logsoftmax 쓰듯이 log를 취해주면 학습이 더 잘됨
            loss = loss_clf(torch.log(out),y)
            loss.backward()
            optimizer.step()
            
            acc = accuracy(out.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss
            torch.cuda.empty_cache()
        logs_clf['acc'] = accs.avg.detach().cpu()
        logs_clf['loss'] = losses.avg.detach().cpu()
        if test_loader:
            logs_clf['val_loss'],logs_clf['val_acc'] = test_model(clf,test_loader,loss_clf,device,best_acc,save_dir,save_model,True)
            if best_acc<logs_clf['val_acc']:
                best_acc = logs_clf['val_acc']

        liveloss_tr.update(logs_clf)
        liveloss_tr.send()  
    
    clf.cpu()
    
    return clf, logs_clf

def test_model_multiple(model, test_loader, loss_list, loss_weights, device, 
                        best_acc=0.0,save_dir = "../results/",save_model = "ckpt.pth"):
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    
    for batch_idx, (x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        p_y = model(x)
        
        loss = 0.0
        for i in range(len(loss_list)):
            loss += loss_list[i](p_y,y) * loss_weights[i]
        
        acc = accuracy(p_y.detach().cpu(), y.detach().cpu())
    
        losses.update(loss,x.size(0))
        accs.update(acc[0],x.size(0))
        del x,y,p_y, loss
        torch.cuda.empty_cache()
#         print(acc)
    if accs.avg>best_acc:
        torch.save(model.state_dict(),os.path.join(save_dir, save_model))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu()

def test_model(model, test_loader, criterion, device, best_acc=0.0,save_dir = "../results/",save_model = "ckpt.pth",pred_prob = False):
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    for batch_idx, (x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        p_y = model(x)
        if pred_prob:
            p_y = torch.log(p_y)
        loss = criterion(p_y,y)
        
        acc = accuracy(p_y.detach().cpu(), y.detach().cpu())
    
        losses.update(loss,x.size(0))
        accs.update(acc[0],x.size(0))
        del x,y,p_y, loss, acc
        torch.cuda.empty_cache()
#         print(acc)
    if accs.avg>best_acc:
        torch.save(model.state_dict(),os.path.join(save_dir, save_model))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu()


