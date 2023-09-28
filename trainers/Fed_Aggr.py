# ========================================================================
# 
# Contains FedAvg and FedEntropyAvg (entropy based aggregation of models )
# 
# ========================================================================
 

import copy
import torch
from torch import nn
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from lib.utils import DatasetSplit, calculate_avg_entropy


def FedAvg(w, global_w=None, size_arr=None):
    w_avg = {}
    for k in w[0].keys():
      w_avg[k] = torch.zeros(w[0][k].size())
      
    # Prepare p 
    if size_arr is not None:
      total_num = np.sum(size_arr)
      size_arr = np.array([float(p)/total_num for p in size_arr])*len(size_arr)
    else:
      size_arr = np.array([1.0]*len(size_arr))

    if global_w is not None:
      for k in w_avg.keys():
          for i in range(0, len(w)):
            grad = w[i][k]
            grad_norm = torch.norm(grad, p=2) / torch.norm(global_w[k], p=2) 
            w_avg[k] += size_arr[i]*grad / grad_norm   
          w_avg[k] = torch.div(w_avg[k], len(w))  
    else:
      for k in w_avg.keys():
          for i in range(0, len(w)):
            w_avg[k] += size_arr[i]*w[i][k] 
          w_avg[k] = torch.div(w_avg[k], len(w))
          
    return w_avg

def FedEntropyAvg(net, w, global_w=None, server_data=None, server_id=None, size_arr=None):
    w_avg = {}
    for k in w[0].keys():
      w_avg[k] = torch.zeros(w[0][k].size())


    server_train_dataset = DataLoader(DatasetSplit(server_data, server_id), batch_size=100, shuffle=False)
    server_dataset_image = [images for images, labels in server_train_dataset]

    entropy_list = [calculate_avg_entropy(net, client_w, server_dataset_image) for client_w in w]
    entropy_list = np.array(entropy_list)

    if size_arr is None:
      size_arr = np.array([1.0]*len(size_arr))
    else:
      size_arr = np.array(size_arr)

    mult_arr = np.true_divide(size_arr, entropy_list)
    mul_sum = np.sum(mult_arr)
    mult_arr = np.array([float(p)/mul_sum for p in mult_arr])*len(mult_arr)

    if global_w is not None:
      for k in w_avg.keys():
          for i in range(0, len(w)):
            grad = w[i][k]
            grad_norm = torch.norm(grad, p=2) / torch.norm(global_w[k], p=2) 
            w_avg[k] += mult_arr[i]*grad / grad_norm   
          w_avg[k] = torch.div(w_avg[k], len(w))  
    else:
      for k in w_avg.keys():
          for i in range(0, len(w)):
            w_avg[k] += mult_arr[i]*w[i][k] 
          w_avg[k] = torch.div(w_avg[k], len(w))
          
    return w_avg
