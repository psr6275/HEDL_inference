import numpy as np
import math
import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets.cifar as cifar
from torchvision import datasets, transforms

from livelossplot import PlotLosses
import os

from .train_utils import test_binary_model
from .eval_utils import AverageVarMeter

import torch.nn.functional as F


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


def make_MIA_loader_with_target(st_model, dataloader,device,batch_size=128,shuffle=False):
#     ori_in = []
    ori_pred = []
    ori_target = []
    ori_ohy = [] #one hot y
    st_model.to(device).eval()
    with torch.no_grad():
        for x,y in dataloader:
            ori_in.append(x)
            pred = st_model(x.to(device)).detach().cpu()
            ori_pred.append(pred)
            ori_target.append(y)
            ori_ohy.append(F.one_hot(y,num_classes=10))
            del x,pred,y
    st_model.cpu()
    ori_ohy = torch.cat(ori_ohy,dim=0)
    ori_pred = torch.cat(ori_pred,dim=0)
    data = torch.cat((ori_pred,ori_ohy),dim=1)
    labels = torch.cat(ori_target,dim=0)
    
    dataloader_ = DataLoader(TensorDataset(data,labels),batch_size=batch_size, shuffle=shuffle)
    return dataloader_
    
def prepare_MIAattack_loader(st_model, st_trainloader, st_validloader, device, batch_size=128, shuffle = True):
    """
        st_trainloader: data loader used for training st_model
        st_validloader: data loader not used for training
        assume that data laoders have hard labels
        use fixed loader for mia attack 
    """
    ori_ohy = [] 
#     ori_out = []
    ori_pred = []
    st_model.to(device)
    st_model.eval()
    with torch.no_grad():
        for x, y in st_trainloader:
            ori_ohy.append(F.one_hot(y,num_classes=10))
#             ori_out.append(F.one_hot(y,num_classes=10))
            pred = st_model(x.to(device)).detach().cpu()
            ori_pred.append(pred)
            del x, pred,y
        for x, y in st_validloader:
            ori_ohy.append(F.one_hot(y,num_classes=10))
            pred = st_model(x.to(device)).detach().cpu()
            ori_pred.append(pred)
            del x, pred
    
    st_model.cpu()
    ori_ohy = torch.cat(ori_ohy,dim=0)
    ori_pred = torch.cat(ori_pred,dim=0)
    data = torch.cat((ori_pred,ori_ohy),dim=1)
    print("data shape:", data.shape)
    
    num_tr = _data_num(st_trainloader)
    num_val = _data_num(st_validloader)
    labels = torch.cat((torch.ones(num_tr), torch.zeros(num_val)))
    print("number of data for training and valid:", num_tr, num_val)
    
    mia_loader = DataLoader(TensorDataset(data, labels), batch_size = batch_size, shuffle=shuffle)
    return mia_loader

def _data_num(dataloader):
    if type(dataloader.dataset) == torch.utils.data.dataset.Subset:
        return len(dataloader.dataset.indices)
    elif type(dataloader.dataset) in [cifar.CIFAR100, cifar.CIFAR10]:
        return len(dataloader.dataset.targets)
    else:
        print("not implemented yet")
        return
    
def train_mia_model(mia_att_model, st_model, mia_loader, criterion, optimizer, epochs, device, mia_testloader = None, save_dir = "../results", save_model = "cifar_mia_model.pth"):
    mia_att_model.to(device)
    logs_clf = {}
    best_acc = 0.0
    liveloss_tr = PlotLosses()
    
    for epoch in range(epochs):
        mia_att_model.train()
        for x,y in mia_loader:
            x = x.to(device)
            y = y.to(device)
            
            mia_att_model.zero_grad()
            out = mia_att_model(x)
            loss = criterion(out.flatten(), y)
            
            loss.backward()
            optimizer.step()
            
            del out, x, y, loss
            torch.cuda.empty_cache()
        logs_clf['loss'], logs_clf['acc']= test_binary_model(mia_att_model, mia_loader, criterion, device, 100.0, save_dir, save_model)
        if mia_testloader is not None:
            logs_clf['val_loss'], logs_clf['val_acc']= test_binary_model(mia_att_model, mia_testloader, criterion, device, 0.0, save_dir, save_model)
        liveloss_tr.update(logs_clf)
        liveloss_tr.send()
    return mia_att_model, logs_clf
    
"""
Systematic Evaluation of Privacy Risks of Machine Learning Models
"""

class black_box_benchmarks(object):
    
    def __init__(self, shadow_train_performance, shadow_test_performance, 
                 target_train_performance, target_test_performance, num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''
        self.num_classes = num_classes
        
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)
        
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
        
    
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)
    
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)
    
    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre
    
    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        return
    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)
        mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
        return
    

    
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        if (all_methods) or ('correctness' in benchmark_methods):
            self._mem_inf_via_corr()
        if (all_methods) or ('confidence' in benchmark_methods):
            self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('entropy' in benchmark_methods):
            self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)

        return