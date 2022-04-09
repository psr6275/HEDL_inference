import torch 
import numpy as np
import matplotlib.pyplot as plt

def maxclass_hist(data_loader, model, device, plt_title = None,bins=10, 
                  return_val = False, clipping =False, clip_vals = [0.0,1.0]):
    model.to(device).eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader,0):
            x = data[0].to(device)
            outs = model(x)
            if i==0:
                max_vals = outs.cpu().detach().numpy()
            else:
                max_vals = np.vstack((max_vals, outs.cpu().detach().numpy()))
            del data, outs,x
    max_vals = np.max(max_vals,axis=1)
    if clipping:
        max_vals = np.clip(max_vals, clip_vals[0], clip_vals[1])
    if plt_title:
        plt.title(plt_title)
    plt.hist(max_vals, bins=bins)
    plt.show()
    model.cpu()
    if return_val:
        return max_vals    

def plot_individual_prediction(batch_x, batch_y, device, net1, net2 = None):
    if net2 is None:
        plot_prediction(batch_x, batch_y, device, net1)
    else:
        net1.to(device)
        net2.to(device)
        net1.eval()
        net2.eval()
        with torch.no_grad():
            x = batch_x.to(device)
            pred1 = net1(x).detach().cpu().numpy()
            pred2 = net2(x).detach().cpu().numpy()
            del x
        net1.cpu()
        net2.cpu()
        for i in range(len(pred1)):
            plt.subplot(1, 2, 1)
            plt.bar(range(10),pred1[i])
            plt.title(str(batch_y[i]))
            plt.subplot(1, 2, 2)
            plt.bar(range(10),pred2[i])
            plt.title(str(batch_y[i]))
            plt.show()
            
def plot_prediction(batch_x, batch_y, device, net):
    net.to(device)
    net.eval()
    with torch.no_grad():
        x = batch_x.to(device)
        pred = net(x).detach().cpu().numpy()
        del x
    net.cpu()
    for i in range(len(pred)):
        plt.bar(range(10), pred[i])
        plt.title(str(batch_y[i]))
        plt.show()    
    
def prediction_hist(data_loader, model, device, plt_title = None,bins=10):
    model.to(device).eval()
    max_idxs = torch.tensor([])
    with torch.no_grad():
        for data in data_loader:
            outs = model(data[0].to(device))
            _,idxs = outs.cpu().detach().max(axis=1)
            max_idxs = torch.cat((max_idxs,idxs))
            del data, outs,idxs
    
    if plt_title:
        plt.title(plt_title)
    plt.hist(max_idxs.numpy().flatten(), bins=bins)
    plt.show()
    model.cpu()
    
class AverageVarMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.sum2 = 0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val = val
#         print(val)
        self.sum2 += (val**2)*n
        self.sum +=val*n
        self.count +=n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    '''Compute the top1 and top k accuracy'''
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res

def accuracy_b(output, target, thres = 0.5):
    '''Compute the binary accuracy'''
    assert output.ndim == 1 and target.size() == output.size()
    y_prob = output>thres 
    return (target == y_prob).sum().item() / target.size(0)

def correspondence_score(output1, output2):
    _, pred1 = torch.max(output1.data, dim=1)
    _, pred2 = torch.max(output2.data, dim=1)
    correct = pred1.eq(pred2)
    return 100*correct.view(-1).float().sum(0)/pred1.size(0)
