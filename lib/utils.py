import copy
import pdb
import os
import pickle  

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn import metrics
from scipy.stats import mode
from scipy.stats import entropy
from scipy.stats import entropy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import json
from collections import defaultdict

def test_img(net_g, datatest, idxs, reweight=None, dataset_type=None):
    net_g.eval()
    test_loss = 0
    correct = 0
    cnt = 0.0
    
    data_loader = DataLoader(DatasetSplit(datatest, idxs), batch_size=1024, shuffle=False)
    l = len(data_loader)
    net_g = net_g
    with torch.no_grad():
      for idx, (data, target) in enumerate(data_loader):

        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        target = target.data.view_as(y_pred)
        correct += y_pred.eq(target).long().cpu().sum()
        cnt += len(data)

    test_loss /= cnt
    accuracy = 100.00 * correct / cnt
    print('{:5s}: loss = {:5.4f}  |  Accuracy = {}/{} ({:4.2f}%)'.format(dataset_type, test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy.numpy(), test_loss

def test_data(net_glob, dataset, ids, dataset_type=None):
    acc, loss = test_img(net_glob, dataset, ids, dataset_type=dataset_type)
    #q.put([acc, loss])
    return [acc, loss]

def test_net_util(net_glob, tag, dataset_train, dataset_test, train_ids):

    [acc_train, loss_train] = test_data(net_glob, dataset_train, train_ids, dataset_type='Train')
    [acc_test, loss_test] = test_data(net_glob, dataset_test, range(len(dataset_test)), dataset_type='Test')

    return [acc_train, loss_train], [acc_test,  loss_test]

def log_test_net(logger, acc_dir, net_glob, tag, iters, dataset_train, dataset_test, train_ids):
    [acc_train, loss_train], [acc_test,  loss_test] = test_net_util(net_glob, tag=tag, dataset_train=dataset_train, dataset_test=dataset_test, train_ids = train_ids)

    if iters==0:
        open(os.path.join(acc_dir, tag+"_train_acc.txt"), "w")
        open(os.path.join(acc_dir, tag+"_test_acc.txt"), "w")
        open(os.path.join(acc_dir, tag+"_test_loss.txt"), "w")

    with open(os.path.join(acc_dir, tag+"_train_acc.txt"), "a") as f:
        f.write("%d %f\n"%(iters, acc_train))
    with open(os.path.join(acc_dir, tag+"_test_acc.txt"), "a") as f:
        f.write("%d %f\n"%(iters, acc_test))
    with open(os.path.join(acc_dir, tag+"_test_loss.txt"), "a") as f:
        f.write("%d %f\n"%(iters, loss_test))
      
    if "SWA" not in tag:
        logger.loss_train_list.append(loss_train)
        logger.train_acc_list.append(acc_train)

        logger.loss_test_list.append(loss_test)
        logger.test_acc_list.append(acc_test)

    else:
        if tag =="SWAG":
            logger.swag_train_acc_list.append(acc_train)
            logger.swag_test_acc_list.append(acc_test)            
        else:
            logger.swa_train_acc_list.append(acc_train)
            logger.swa_test_acc_list.append(acc_test)     

def calculate_avg_entropy(net, client_w, server_dataset_image):
  net.load_state_dict(client_w)
  prob_pred = None
  with torch.no_grad():
    for img in server_dataset_image:
      output = net(img)
      output = F.softmax(output,dim=1)
      if prob_pred is None:
        prob_pred = torch.Tensor(output)
      else:
        prob_pred = torch.cat([prob_pred, torch.Tensor(output)],dim=0)
  entropy = calculate_entropy(prob_pred)
  avg_entropy = np.mean(entropy, axis=0)
  return avg_entropy

def calculate_entropy(probs_tensor):
  entropy = []
  probs_list = np.array(probs_tensor)
  for i in range(len(probs_list)):
    ent_cal = (-probs_list[i, :] * np.log(probs_list[i, :] + 1e-8)).sum()
    entropy.append(ent_cal)
  return np.array(entropy)

def store_model(iter, model_dir, w_glob_org, client_w_list):
    torch.save(w_glob_org, os.path.join(model_dir, "w_org_%d"%iter)) 
    for i in range(len(client_w_list)):
      torch.save(client_w_list[i], os.path.join(model_dir, "client_%d_%d"%(iter, i)))  

def adaptive_schedule(local_ep, total_ep, rounds, adap_ep):
  if rounds<5:
    running_ep = adap_ep
  else:
    running_ep = local_ep
  return running_ep
    
def get_entropy(logits):
    mean_entropy = np.mean([entropy(logit) for logit in logits])
    return mean_entropy

def subsample_data(lab_values,incl_indices,num_sample):
  freq = [0 for i in range(52)]
  for ind in incl_indices:
    freq[lab_values[ind]-10]+=1
  compose_freq = [0 for i in range(52)]
  tot_sample = len(incl_indices)
  for cl in range(52):
    compose_freq[cl] = int(math.ceil(float(freq[cl]*num_sample)/tot_sample))
  new_incl_indices = []
  for ind in incl_indices:
    if compose_freq[lab_values[ind]-10]>0:
      new_incl_indices.append(ind)
      compose_freq[lab_values[ind]-10]-=1
  return new_incl_indices

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):

    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data
    
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label    
        
def get_input_logits(inputs, model, is_logit=False, net_org=None):
    model.eval()
    with torch.no_grad():
      logit = model(inputs).detach()
      if not is_logit:
        logit = F.softmax(logit, dim=1)
       
    logit = logit.cpu().numpy()
    return logit 
    
def temp_softmax(x, axis=-1, temp=1.0):
    x = x/temp
    e_x = np.exp(x - np.max(x)) # same code
    e_x = e_x / e_x.sum(axis=axis, keepdims=True)
    return e_x
    
def temp_sharpen(x, axis=-1, temp=1.0):
    x = np.maximum(x**(1/temp), 1e-8)
    return x / x.sum(axis=axis, keepdims=True)

    
def merge_logits(logits, method, loss_type, temp=0.3, global_ep=1000):
    if "vote" in method:
      if loss_type=="CE":
        votes = np.argmax(logits, axis=-1) 
        logits_arr = mode(votes, axis=1)[0].reshape((len(logits)))
        logits_cond = np.mean(np.max(logits, axis=-1), axis=-1)
      else:  
        logits = np.mean(logits, axis=1)
        logits_arr = temp_softmax(logits, temp=temp) 
        logits_cond = np.max(logits_arr, axis=-1)
    else:
      logits = np.mean(logits, axis=1)
      
      if loss_type=="MSE":
        logits_arr = temp_softmax(logits, temp=1)
        logits_cond = np.max(logits_arr, axis=-1)
      elif "KL" in loss_type: 
        logits_arr = temp_sharpen(logits, temp=temp)   
        logits_cond = np.max(logits_arr, axis=-1)
      else:
        logits_arr = logits
        logits_cond = softmax(logits, axis=-1)
        logits_cond = np.max(logits_cond, axis=-1)    

    return logits_arr, logits_cond  

def weights_init(m): 
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
          torch.nn.init.zeros_(m.bias)
          
class logger():
  def __init__(self, name):
    self.name = name
    self.loss_train_list = []
    self.loss_test_list = []
    
    
    self.train_acc_list = []
    self.test_acc_list = []
    self.val_acc_list = [] 
    self.loss_val_list = []
    
    self.ens_train_acc_list = []
    self.ens_test_acc_list = []
    self.ens_val_acc_list = []
    
    
    self.teacher_loss_train_list = []
    self.teacher_loss_test_list = []
    
    self.swa_train_acc_list=[]
    self.swa_test_acc_list=[]
    self.swa_val_acc_list = []
    
    self.swag_train_acc_list=[]
    self.swag_test_acc_list=[]
    self.swag_val_acc_list = []


