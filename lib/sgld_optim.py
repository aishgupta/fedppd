# ============================
# 
# SGLD optimizer
# 
# ============================


from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch

# References
# repository https://github.com/wangkua1/apd_public/
# https://github.com/JavierAntoran/Bayesian-Neural-Networks

class SGLD(Optimizer):

    def __init__(self, params, lr=required, weight_decay=0.001, datasize=1, addnoise=True):
        
        # print('SGLD optimizer Initialized')
        self.correction = datasize/2
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, addnoise=addnoise)

        super(SGLD, self).__init__(params, defaults)

    def step(self):

        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data,alpha=weight_decay)
                
                if group['addnoise']:
                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1)/(np.sqrt(group['lr'])*self.correction)
                    p.data.add_(0.5 * d_p + langevin_noise,alpha=-group['lr'])
                else:
                    p.data.add_(0.5 * d_p,alpha=-group['lr'])

        return loss