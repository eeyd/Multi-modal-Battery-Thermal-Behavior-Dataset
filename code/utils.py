######################
# Author: Ronghe Chu
######################
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score, recall_score, f1_score
import config

def calculate_discripancy_loss(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))
    
def batch_accuracy(out, labels):
    logits = torch.max(out, 1)[1].data # argmax
    return (labels.long() == logits).sum().item() / len(logits)
    
def calculate_batch_PRF(out, labels):
    logits = torch.max(out, 1)[1].data # argmax
    precision = precision_score(labels.cpu().long(), logits.long(),average='binary')
    recall = recall_score(labels.cpu().long(),logits.long(),average='binary')
    f1 = f1_score(labels.cpu().long(), logits.long(), average='binary')
    return precision, recall, f1

def calculate_batch_PRFS(out, labels):
    precision, recall, fscore, support = score(labels.cpu().long(), out.cpu().max(1)[1].long(), average='binary')
    return precision, recall, fscore, support

def calculate_cross_entropy_loss(scores, label):
    loss = F.cross_entropy(scores, label, config.weight)
    return loss
    
def calculate_cross_entropy_loss_weighted(scores, label):
    loss = F.cross_entropy(scores, label, config.weight_trans)
    return loss
    
def calculate_BCE_with_logits_loss(scores, one_hot_label, weight):
    loss = F.binary_cross_entropy_with_logits(scores, one_hot_label, weight) 
    return loss
    
    
class CE_FocalLoss(nn.Module):
    '''nn.CrossEntropyLoss'''
    def __init__(self, alpha=0.25, gamma=2):
        super(CE_FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.alpha=alpha
    def forward(self, scores, target):
        logp = self.ce(scores, target)
        p = torch.exp(-logp)
        
        loss = self.alpha*(1 - p) ** self.gamma * logp * target.long() + \
               (1-self.alpha)*(p) ** self.gamma * logp * (1-target.long())
        
        loss = torch.mean(loss)  
        return loss


class BCE_FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25,gamma=2):
        super(BCE_FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt)# - \
              #(1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        loss = torch.mean(loss)
        return loss
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, final_reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.final_reduce = final_reduce

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.final_reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


'''
x = np.array([[1, 2],
              [1, 2],
              [1, 1]]).astype(np.float32)
y = np.array([1, 1, 0])
y_one_hot = np.array([[1,0],
                      [1,0],
                      [0,1]])
x = torch.from_numpy(x)
print(F.softmax(x, dim=1))
y = torch.from_numpy(y).long()
y_one_hot = torch.from_numpy(y_one_hot).float()
model1 = CE_FocalLoss()
model2 = BCE_FocalLoss()
model3 = FocalLoss()
loss1 = model1(x,y)
loss2 = model2(x,y_one_hot)
loss3 = model3(x,y_one_hot)
print(loss1)
print(loss2)
print(loss3)
'''

'''
labels = torch.Tensor(np.array([1, 1, 1, 1, 1, 0,0,0,0,0]))
x = np.array([[1, 2],
              [1, 2],
              [1, 2],
              [1, 2],
              [1, 2],
              [2, 1],
              [2, 1],
              [2, 1],
              [2, 1],
              [1, 2]]).astype(np.float32)
out = torch.Tensor(x)
acc = batch_accuracy(out, labels)

P,R,F,S = calculate_batch_PRFS(out, labels)
print(acc)
print(P)
print(R)
print(F)
print(S)
'''


