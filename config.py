######################
# Author: Ronghe Chu
######################
import numpy as np
import torch
###################################################################
#              Training Setting 
###################################################################
seed = 10
epochs = 50
batch_size = 3
initial_lr = 1e-3
charge_type = ['ch','och','dch','odch'] # charge/ over charge/ discharge/ over discharge
charge_feature = 32

if_gpu = True
if if_gpu:
    device = 'cuda:0'
else:
    device = 'cpu'
    
###################################################################
#              Network Setting 
###################################################################

rnn_layers = 1
volt_features = 64
temp_features = 64
curr_features = 64

img_features = 512
mid_features = 256

type_feature = 32

#fusion = 'element_multiplication'
fusion = 'element_addition'


### Training Setting ###
in_step=6
h_step=4
Sampling_rate = 5


###################################################################
#              Imbalance Strategy
###################################################################
imbalance = False
if imbalance:
    weight = [2.0, 1.0]
    weight = torch.Tensor(weight).to(device)
else:
    weight = [1,1]
    weight = torch.Tensor(weight).to(device)
    


weight_trans  = [1.0, 1.0]
weight_trans = torch.Tensor(weight_trans).to(device)

