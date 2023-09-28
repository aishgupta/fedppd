#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import pdb
from torchvision import datasets, transforms
import os
import glob
from torch.utils.data import Dataset
from PIL import Image

def mnist_iid(dataset, num_users, num_data=50000):

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    server_idx = np.random.choice(all_idxs, 60000-num_data, replace=False)
    all_idxs = list(set(all_idxs) - set(server_idx))
    num_items = int(len(all_idxs)/num_users)
    
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    cnts_dict = {}
    with open("mnist_data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))
    return dict_users, server_idx, cnts_dict

def mnist_noniid(dataset, num_users, num_data=60000, method='step', img_use_frac=1.0):
    """
    Sample non-I.I.D client data from MNIST dataset
    """

    _lst_sample = 10

    if method=='step':
      num_shards, num_imgs = 25, 2000
      idx_shard = [i for i in range(num_shards)]
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      idxs = np.arange(num_shards*num_imgs)
      labels = dataset.targets.numpy()[:num_shards*num_imgs]
      lab_serv = dataset.targets.numpy()
      # sort labels
      idxs_labels = np.vstack((idxs, labels))
      idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
      idxs = idxs_labels[0,:]

      least_idx = np.zeros((num_users, 10, _lst_sample), dtype=np.int)

      for i in range(10):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))

      # divide and assign
      for i in range(num_users):
          rand_set = set(np.random.choice(idx_shard, 2, replace=False))
          idx_shard = list(set(idx_shard) - rand_set)
          for rand in rand_set:
              add_idx = np.array(list(set(idxs[rand*num_imgs:rand*num_imgs + int(num_imgs*img_use_frac)]) ))
              dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)
          dict_users[i] = np.concatenate((dict_users[i], least_idx[i]), axis=0)
      server_idx = list(range(num_shards*num_imgs, 60000))
      server_idx = np.array(server_idx)
    else:
      #code here
      pass

    cnts_dict = {}
    with open("data_mnist_niid_%d_u%d.txt"%(num_data, num_users), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
      label_s = lab_serv[server_idx]
      count_serv = np.array([np.count_nonzero(label_s == j ) for j in range(10)] )
      f.write("Server Labels: %s sum: %d\n"%(" ".join([str(c) for c in count_serv]), sum(count_serv) ))  
    return dict_users, server_idx, cnts_dict

def cifar_iid(dataset, num_users, num_data=40000, method='random'):
    
    labels = np.array(dataset.targets)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    server_idx = np.random.choice(all_idxs, 50000-num_data, replace=False)
    all_idxs = list(set(all_idxs) - set(server_idx))
    num_items = int(len(all_idxs)/num_users)
    
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))

    cnts_dict = {}
    with open("cifar10_data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))

    return dict_users, server_idx, cnts_dict
    
def cifar_noniid(dataset, num_users, num_data=40000, img_use_frac=1.0, method="step"):

    labels = np.array(dataset.targets)
    _lst_sample = 10 
    
    if method=="step":
      
      num_shards = num_users*2
      num_imgs = 50000// num_shards
      idx_shard = [i for i in range(num_shards)]
      
      idxs = np.arange(num_shards*num_imgs)
      # sort labels
      idxs_labels = np.vstack((idxs, labels))
      idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
      idxs = idxs_labels[0,:]
      
      least_idx = np.zeros((num_users, 10, _lst_sample), dtype=np.int)
      for i in range(10):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      
      # divide and assign
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      for i in range(num_users):
          rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
          idx_shard = list(set(idx_shard) - rand_set)
          for rand in rand_set:
              idx_i = list( set(range(rand*num_imgs, rand*num_imgs + int(num_imgs*img_use_frac)))   )
              add_idx = list(set(idxs[idx_i]) - set(server_idx)  )              
              dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)
          dict_users[i] = np.concatenate((dict_users[i], least_idx[i]), axis=0)
 
    elif method == "dir":
      min_size = 0
      K = 10
      y_train = labels
      
      _lst_sample = 2

      least_idx = np.zeros((num_users, 10, _lst_sample), dtype=np.int)
      for i in range(10):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      #least_idx_set = set([])
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      local_idx = np.array([i for i in range(50000) if i not in server_idx and i not in least_idx_set])
      
      N = y_train.shape[0]
      net_dataidx_map = {}
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

      while min_size < 10:
          idx_batch = [[] for _ in range(num_users)]
          # for each class in the dataset
          for k in range(K):
              idx_k = np.where(y_train == k)[0]
              idx_k = [id for id in idx_k if id in local_idx]
              
              np.random.shuffle(idx_k)
              proportions = np.random.dirichlet(np.repeat(0.1, num_users))
              ## Balance
              proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
              proportions = proportions/proportions.sum()
              proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
              idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
              min_size = min([len(idx_j) for idx_j in idx_batch])

      for j in range(num_users):
          np.random.shuffle(idx_batch[j])
          dict_users[j] = idx_batch[j]  
          dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          

    cnts_dict = {}
    with open("cifar10_data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
   
    return dict_users, server_idx, cnts_dict

def cifar100_iid(dataset, num_users, num_data=50000, method='random'):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    labels = np.array(dataset.targets)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    server_idx = np.random.choice(all_idxs, 50000-num_data, replace=False)
    all_idxs = list(set(all_idxs) - set(server_idx))
    num_items = int(len(all_idxs)/num_users)
    
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))

    cnts_dict = {}
    with open("cifar100_data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))

    return dict_users, server_idx, cnts_dict
    
def cifar100_noniid(dataset, num_users, num_data=50000, img_use_frac=1.0, method="step", n_class=100, n_class_per_user=20, lst_sample=2):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """

    labels = np.array(dataset.targets)
    _lst_sample = lst_sample 
    
    if method=="step":
      
      num_shards = num_users*n_class_per_user
      num_imgs = 50000// num_shards
      idx_shard = [i for i in range(num_shards)]
      
      idxs = np.arange(num_shards*num_imgs)
      # sort labels
      idxs_labels = np.vstack((idxs, labels))
      idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
      idxs = idxs_labels[0,:]
      
      least_idx = np.zeros((num_users, n_class, _lst_sample), dtype=np.int)
      for i in range(n_class):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      
      # divide and assign
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      for i in range(num_users):
          rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
          idx_shard = list(set(idx_shard) - rand_set)
          for rand in rand_set:
              idx_i = list( set(range(rand*num_imgs, rand*num_imgs + int(num_imgs*img_use_frac)))   )
              add_idx = list(set(idxs[idx_i]) - set(server_idx)  )              
              dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)
          dict_users[i] = np.concatenate((dict_users[i], least_idx[i]), axis=0)
 
    elif method == "dir":
      min_size = 0
      K = 10
      y_train = labels
      
      _lst_sample = 2

      least_idx = np.zeros((num_users, 10, _lst_sample), dtype=np.int)
      for i in range(10):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      #least_idx_set = set([])
      server_idx = np.random.choice(list(set(range(50000))-least_idx_set), 50000-num_data, replace=False)
      local_idx = np.array([i for i in range(50000) if i not in server_idx and i not in least_idx_set])
      
      N = y_train.shape[0]
      net_dataidx_map = {}
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

      while min_size < 10:
          idx_batch = [[] for _ in range(num_users)]
          # for each class in the dataset
          for k in range(K):
              idx_k = np.where(y_train == k)[0]
              idx_k = [id for id in idx_k if id in local_idx]
              
              np.random.shuffle(idx_k)
              proportions = np.random.dirichlet(np.repeat(0.1, num_users))
              ## Balance
              proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
              proportions = proportions/proportions.sum()
              proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
              idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
              min_size = min([len(idx_j) for idx_j in idx_batch])

      for j in range(num_users):
          np.random.shuffle(idx_batch[j])
          dict_users[j] = idx_batch[j]  
          dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          

    cnts_dict = {}
    with open("data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(n_class)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
   
    return dict_users, server_idx, cnts_dict
