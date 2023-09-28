# ===========================================
# 
# Contains functions to:
# * distill PPD at client
# * distillation based aggregation at server
# 
# References:
# https://github.com/hongyouc/FedBE
# 
# ===========================================

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random, pdb, os
import torch.nn.functional as F
import copy
from torchvision import datasets, transforms

from scipy.stats import mode
from lib.utils import *
from sklearn.utils import shuffle
from PIL import Image
from lib.sgld_optim import SGLD

import torch.multiprocessing as mp
from trainers.swa import SWA 

class AverageMeter(object):
  """
  Computes and stores the average and current value
  Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
  """
  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
   
class LocalTSUpdate(object):
    def __init__(self, args, device, dataset=None, idxs=None, test=(None, None), num_per_cls=None):
        self.args = args
        self.device = device
        self.num_per_cls = num_per_cls
        if args.dataset=='femnist':
          self.N_train = len(dataset)
        else:
          self.N_train = len(idxs)
        # print('Training client with ', self.N_train, ' amount of data')
        
        #WARNING: drop last batch to avoid issue with BN training in networks with BN layers
        if args.model=='resnet':
          drop_last=True
        else:
          drop_last=False
        
        if args.dataset=='femnist':
          self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        else:
          self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=drop_last)
        (self.test_dataset, self.test_ids) = test
        (self.train_dataset, self.user_train_ids) = (dataset, idxs)
        
    def train(self, tnet, snet, local_eps, t_lr, t_wd, s_lr, s_wd, gamma, stepsiz): 
      tnet.cpu()
      snet.cpu()
      tnet.to(self.device)
      tnet.eval()
      snet.to(self.device)
      snet.eval()

      teach_optim = SGLD(params=tnet.parameters(), lr=t_lr, weight_decay=t_wd, datasize=self.N_train, addnoise=True)
      teach_sched = torch.optim.lr_scheduler.StepLR(teach_optim, step_size=stepsiz, gamma=gamma)

      stud_optim = torch.optim.SGD(params=snet.parameters(), lr=s_lr, weight_decay=s_wd)

      teach_epoch_loss = []
      stud_epoch_loss = []
      acc = 0.0

      num_model = 0
      cnt = 0
      for iter in range(local_eps):
          tnet.train()
          snet.eval()
          ep_loss = AverageMeter()
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.device), labels.to(self.device)
            tnet.zero_grad()
            out = tnet(images)
            loss = F.cross_entropy(out, labels, reduction='mean')
            loss.backward()
            teach_optim.step()
            pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
            err = pred.ne(labels.data).sum()
            err = (err+0.0)/len(images)
            if self.args.verbose and batch_idx % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\t Teacher Loss: {:.6f} Error: {:.6f}'.format(iter, batch_idx * len(images), 
                len(self.ldr_train.dataset),
                100. * batch_idx / len(self.ldr_train), loss.item(), err))

            ep_loss.update(loss.item(), images.shape[0])

          teach_sched.step()
          teach_epoch_loss.append(ep_loss.avg)
          snet.train()
          tnet.eval()

          ep_loss = AverageMeter()
          for batch_idx, (imag_train, labels) in enumerate(self.ldr_train):
            imag_train, labels = imag_train.to(self.device), labels.to(self.device)
            n_data = torch.normal(mean=torch.zeros(imag_train.size()),std=0.001).to(self.device)
            images = imag_train + n_data

            snet.zero_grad()
            out1 = snet(images)
            out2 = tnet(images)
            log_probs_st = F.log_softmax(out1,dim=1)
            log_probs_teach = F.softmax(out2,dim=1)
            loss = -torch.mean(torch.sum(torch.mul(log_probs_st,log_probs_teach),axis=1))
            loss.backward()
            stud_optim.step()
            pred = out1.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
            err = pred.ne(labels.data).sum()
            err = (err+0.0)/len(images)
            if self.args.verbose and batch_idx % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\t Student Loss: {:.6f} Error: {:.6f}'.format(iter, batch_idx * len(images), 
                len(self.ldr_train.dataset),
                100. * batch_idx / len(self.ldr_train), loss.item(), err))
            ep_loss.update(loss.item(), images.shape[0])
          stud_epoch_loss.append(ep_loss.avg)    
      tnet.eval()
      snet.eval() 

      tnet = tnet.cpu()
      snet = snet.cpu()    
      return tnet, snet, teach_epoch_loss, stud_epoch_loss

class ServerUpdate(object):
    def __init__(self, args, device, dataset=None, 
                 server_dataset=None, server_idxs=None, 
                 test=(None, None), 
                 w_org=None, base_teachers=None, server_lr=1e-3):
                 
        self.args = args
        self.device = device
        self.loss_type = args.loss_type
        self.loss_func = nn.KLDivLoss() if self.loss_type =="KL" else nn.CrossEntropyLoss()
        self.selected_clients = []
        self.server_lr=server_lr
        
        self.server_data_size = len(server_idxs)
        self.ldr_train = DataLoader(DatasetSplit(dataset, server_idxs), batch_size=args.server_bs, shuffle=False)
        self.test_dataset = DataLoader(test[0], batch_size=self.args.server_bs, shuffle=False) 
        self.aum_dir = os.path.join(self.args.log_dir, "aum")  
        
        server_train_dataset = DataLoader(DatasetSplit(server_dataset, server_idxs), batch_size=self.args.server_bs, shuffle=False)
        self.server_train_dataset = [images for images, labels in server_train_dataset]
        
        self.w_org = w_org
        self.base_teachers = base_teachers
        
        # Get one batch for testing
        (self.eval_images, self.eval_labels) = next(iter(self.ldr_train))

    def get_ensemble_logits(self, teachers, inputs, method='mean', global_ep=1000):
        logits = np.zeros((len(teachers), len(inputs), self.args.num_classes))
        for i, t_net in enumerate(teachers):
          logit = get_input_logits(inputs, t_net.to(self.device), is_logit = False) #Disable res
          logits[i] = logit
          
        logits = np.transpose(logits, (1, 0, 2)) # batchsize, teachers, 10
        logits_arr, logits_cond = merge_logits(logits, method, self.args.loss_type, temp=self.args.temp, global_ep=global_ep)
        batch_entropy = get_entropy(logits.reshape((-1, self.args.num_classes)))
        return logits_arr, batch_entropy

    def eval_ensemble(self, teachers, dataset):
        acc = 0.0
        cnt = 0
        
        if self.args.soft_vote:
          num_votes_list, soft_vote = get_aum(self.args, teachers, dataset)
          for batch_idx, (_, labels) in enumerate(dataset):
              logits = soft_vote[batch_idx]
              logits=np.argmax(logits, axis=-1)
              acc += np.sum(logits==labels.numpy())
              cnt += len(labels)            

        else:
          for batch_idx, (images, labels) in enumerate(dataset):
              images = images.to(self.device)
              logits, _ = self.get_ensemble_logits(teachers, images, method='mean', global_ep=1000)
              
              '''
              if self.args.logit_method != "vote":
                logits=np.argmax(logits, axis=-1)
              '''
              logits=np.argmax(logits, axis=-1)
              acc += np.sum(logits==labels.numpy())
              cnt += len(labels)

        return float(acc)/cnt*100.0

    def loss_wrapper(self, log_probs, logits, labels):        
        # Modify target logits
        if self.loss_type=="CE":
          if self.args.logit_method != "vote":
            logits = np.argmax(logits, axis=-1)
          acc_cnt=np.sum(logits==labels)
          cnt=len(labels)
          logits = torch.Tensor(logits).long().to(self.device) 
            
        else:  
          acc_cnt=np.sum(np.argmax(logits, axis=-1)==labels)
          cnt=len(labels)
          logits = torch.Tensor(logits).to(self.device) 


        # For loss function
        if self.args.use_oracle:
          loss = nn.CrossEntropyLoss()(log_probs, torch.Tensor(labels).long().to(self.device))
        else:      
          tol = 1e-8
          if "KL" in self.loss_type:
            log_probs = F.softmax(log_probs, dim=-1)
            if self.loss_type== "reverse_KL":
              P = log_probs + tol
              Q = logits + tol       
            else:
              P = logits + tol
              Q = log_probs + tol
            
            one_vec = (P * (P.log() - torch.Tensor([0.1]).to(self.device).log()))
            loss = (P * (P.log() - Q.log())).mean()
          else:
            loss = self.loss_func(log_probs, logits)
            
        return loss, acc_cnt, cnt

    def record_teacher(self, ldr_train, net, teachers, global_ep, log_dir=None, probe=True, resample=False):
        entropy = []  
        ldr_train = []

        acc_per_teacher = np.zeros((len(teachers)))
        conf_per_teacher = np.zeros((len(teachers)))
        teacher_per_sample = 0.0
        has_correct_teacher_ratio = 0.0
        
        num = self.server_data_size
        if "cifar" in self.args.dataset:
          #includes cifar10 and cifar100
          imgsize = 32 
        elif "mnist" in self.args.dataset:  
          #includes mnist and femnist  
          imgsize = 28
          
        channel = 1 if (self.args.dataset == "mnist" or self.args.dataset == "femnist") else 3        
        all_images = np.zeros((num, channel, imgsize, imgsize))
        all_logits = np.zeros((num, self.args.num_classes))
        all_labels = np.zeros((num))
        cnt = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            logits, batch_entropy = self.get_ensemble_logits(teachers, images.to(self.device), method='mean', global_ep=global_ep)
            entropy.append(batch_entropy)
            
            all_images[cnt:cnt+len(images)] = images.numpy()
            all_logits[cnt:cnt+len(images)] = logits
            all_labels[cnt:cnt+len(images)] = labels.numpy()
            cnt+=len(images)

        ldr_train = (all_images, all_logits, all_labels)
        #=============================
        # If args.soft_vote = True: 
        #    soft_vote from experts
        # Else: 
        #    just mean of all logits
        #=============================
        if not probe:
          return ldr_train, 0.0, 0.0
        else:
          test_acc = self.eval_ensemble(teachers, self.test_dataset)
          train_acc = self.eval_ensemble(teachers, self.ldr_train)
          
          plt.plot(range(len(teachers)), acc_per_teacher, marker="o", label="Acc")
          plt.plot(range(len(teachers)), conf_per_teacher, marker="o", label="Confidence")
          plt.plot(range(len(teachers)), conf_per_teacher - acc_per_teacher, marker="o", label="Confidence - Acc")
          plt.ylim(ymax = 1.0, ymin = -0.2)
          plt.title("Round %d, correct teacher/per sample %.2f, upperbound correct %.1f percentage"%(global_ep, teacher_per_sample,has_correct_teacher_ratio*100.0))
          plt.legend(loc='best')
          plt.savefig(os.path.join(log_dir, "acc_per_teacher_%d.png"% global_ep))       
          plt.clf()        

          return ldr_train, train_acc, test_acc

    def set_opt(self, net):
        #base_opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00001)
        base_opt = torch.optim.SGD(net.parameters(), lr=self.server_lr, momentum=0.9, weight_decay=self.args.weight_decay)
        #base_opt = torch.optim.Adam(net.parameters(), lr=self.server_lr, weight_decay=self.args.weight_decay)
        if self.args.use_SWA:
            self.optimizer = SWA(base_opt, swa_start=500, swa_freq=25, swa_lr=None)
        else:
            self.optimizer = base_opt
    
    def train(self, net, teachers, log_dir, global_ep, server_dataset=None):
        #======================Record teachers========================
        self.set_opt(net)
        
        to_probe = True if global_ep%self.args.log_ep==0 else False
        ldr_train = []
        ldr_train, train_acc, test_acc = self.record_teacher(ldr_train, net, teachers, global_ep, log_dir, probe=to_probe)
        (all_images, all_logits, all_labels) = ldr_train 
        #======================Server Train========================
        print("Start server training...")
        net.to(self.device)        
        net.train()

        epoch_loss = []
        acc = 0
        cnt = 0
        
        step = 0
        train_ep = self.args.server_ep
        for iter in range(train_ep):
            all_ids = list(range(len(all_images)))
            np.random.shuffle(all_ids)
            
            batch_loss = []    
            for batch_idx in range(0, len(all_images), self.args.server_bs):
                ids = all_ids[batch_idx:batch_idx+self.args.server_bs]                
                images = all_images[ids]
                images = torch.Tensor(images).to(self.device)   
                logits = all_logits[ids]
                labels = all_labels[ids]

                net.zero_grad()
                log_probs = net(images)
                
                loss, acc_cnt_i, cnt_i = self.loss_wrapper(log_probs, logits, labels)
                acc+=acc_cnt_i
                cnt+=cnt_i                
                loss.backward()
                
                self.optimizer.step()
                step+=1
            
                if batch_idx == 0 and iter%5==0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        val_acc = float(acc)/cnt*100.0   
        net_glob = copy.deepcopy(net)
        
        if self.args.use_SWA:
          self.optimizer.swap_swa_sgd()
          if "resnet" in self.args.model:
            self.optimizer.bn_update(self.ldr_train, net, device=self.device)
            
        net = net.cpu()   
        w_glob_avg = copy.deepcopy(net.state_dict())
        w_glob = net_glob.cpu().state_dict()
        
        print("Ensemble Acc Train %.2f Val %.2f Test %.2f mean entropy %.5f"%(train_acc, val_acc, test_acc, 0.0))    
        return w_glob_avg, w_glob, train_acc, val_acc, test_acc, sum(epoch_loss) / len(epoch_loss), 0.0
