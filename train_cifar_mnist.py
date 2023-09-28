# =============================
# 
# Main file to run below methods on datasets cifar10, cifar100, mnist:
# * FedPPD
# * FedPPD+Entropy
# * FedPPD+Distill
# 
# =============================

import copy
import json
import os
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms

from lib.dataset_sampler import *
from lib.cnn_models import CNN_Cifar, CNN_Cifar_Stud, CNN_Mnist, CNN_Mnist_Stud
from trainers.Fed_Aggr import FedAvg, FedEntropyAvg
from trainers.ClientServUpdate import LocalTSUpdate, ServerUpdate
from trainers.swag import SWAG_server
from lib.options import args_parser
from lib.utils import *

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    args.log_dir = os.path.join(args.log_dir)   

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)  

    args.acc_dir = os.path.join(args.log_dir, "acc")
    if not os.path.exists(args.acc_dir):
        os.makedirs(args.acc_dir)  
        
    model_dir = os.path.join(args.log_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    

    # data augmentation
    transform_mnist       = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize((0.1307,), (0.3081,))])     

    transform_cifar_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_cifar_val   = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # load dataset and split it among clients
    if args.dataset == 'mnist':
        args.num_classes = 10
        dataset_train    = datasets.MNIST('data/mnist/', train=True, download=True, transform=transform_mnist)
        dataset_test     = datasets.MNIST('data/mnist/', train=False, download=True, transform=transform_mnist)
        dataset_eval     = datasets.MNIST('data/mnist/', train=True, download=True, transform=transform_mnist)        
        
        # sample users
        if args.iid:
            dict_users, server_id, cnts_dict = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users, server_id, cnts_dict  = mnist_noniid(dataset_train, args.num_users, method=args.split_method, img_use_frac=args.img_use_frac)
    
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        dataset_train    = datasets.CIFAR10('data/cifar10', train=True , download=True, transform=transform_cifar_train)
        dataset_test     = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=transform_cifar_val)
        dataset_eval     = datasets.CIFAR10('data/cifar10', train=True , transform=transform_cifar_val, target_transform=None, download=True)
        
        if args.config_file is not None:
            config_dict = json.load(args.config_file)

            dict_users = config_dict["DATA_DISTRIBUTION"]
            server_id  = config_dict["SER_DATA"]
            cnts_dict  = config_dict["CNT_LAB"]
    
        else:
            if args.iid:
                dict_users, server_id, cnts_dict = cifar_iid(dataset_train, args.num_users, num_data=args.num_data)
            else:
                dict_users, server_id, cnts_dict = cifar_noniid(dataset_train, args.num_users, num_data=args.num_data, img_use_frac=args.img_use_frac,method=args.split_method)
    
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        dataset_train    = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=transform_cifar_train)
        dataset_test     = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=transform_cifar_val)
        dataset_eval     = datasets.CIFAR100('data/cifar100', train=True, transform=transform_cifar_val, target_transform=None, download=True)
        
        if args.iid:
            dict_users, server_id, cnts_dict = cifar100_iid(dataset_train, args.num_users, num_data=args.num_data)
        else:
            dict_users, server_id, cnts_dict = cifar100_noniid(dataset_train, args.num_users, num_data=args.num_data, img_use_frac=args.img_use_frac, n_class=100, n_class_per_user=20, lst_sample=2)
    
    else:
        exit('Error: unsupported dataset')
        

    with open(os.path.join(args.log_dir, "args.txt"), "w") as f:
        for arg in vars(args):
            print (arg, getattr(args, arg), file=f)        
        
    train_ids = set()
    for u,v in dict_users.items():
        train_ids.update(v)
    train_ids = list(train_ids)     

    if args.dataset == "cifar100":
        teacher_model = torchvision.models.resnet18(pretrained=True)
        teacher_model.fc = torch.nn.Linear(512, 100)
        student_model = torchvision.models.resnet34(pretrained=True)
        student_model.fc = torch.nn.Linear(512, 100)

    elif args.model=='cnn' and args.dataset=='cifar10':
        tmodel_class = CNN_Cifar
        smodel_class = CNN_Cifar_Stud
        
        teacher_model = tmodel_class()
        student_model = smodel_class()
    
    elif args.model=='cnn' and args.dataset=='mnist':
        tmodel_class = CNN_Mnist
        smodel_class = CNN_Mnist_Stud
        
        teacher_model = tmodel_class()
        student_model = smodel_class()
    
    else:
        exit('Error: unrecognized model')      
        
    print('\nGlobal Teacher Model', teacher_model)
    print('\nGlobal Student Model', student_model)

    # copy weights
    teacher_wt = teacher_model.state_dict()
    student_wt = student_model.state_dict()

    if args.dataset=="cifar100":
        torch.nn.init.xavier_uniform_(teacher_model.fc.weight)
        torch.nn.init.zeros_(teacher_model.fc.bias)
        torch.nn.init.xavier_uniform_(student_model.fc.weight)
        torch.nn.init.zeros_(student_model.fc.bias)
    else:
        teacher_model.apply(weights_init) 
        student_model.apply(weights_init)   


    def client_ts_train_func(q, tnet, snet, t_lr, s_lr, t_wd, s_wd, t_gamma, t_ss, iters, idx, generator=None):
        device=torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        local_eps = args.local_ep

        local = LocalTSUpdate(args=args, device=device, dataset=dataset_train, idxs=dict_users[idx], 
                                test=(dataset_test, range(len(dataset_test))), num_per_cls=cnts_dict[idx])

        teach_model, stud_model, teach_epoch_loss, stud_epoch_loss = local.train(tnet=tnet.to(device), snet=snet.to(device), local_eps=local_eps, t_lr=t_lr, t_wd=t_wd, s_lr=s_lr, s_wd= s_wd, gamma=t_gamma, stepsiz=t_ss)
        if args.dataset=="cifar100":
            tm = torchvision.models.resnet18()
            tm.fc = torch.nn.Linear(512, 100)
            sm = torchvision.models.resnet34()
            sm.fc = torch.nn.Linear(512, 100)
        else:
            tm = tmodel_class()
            sm = smodel_class()
            
        tm.load_state_dict(teach_model.state_dict())
        sm.load_state_dict(stud_model.state_dict())
        teach_ep_loss_copy = teach_epoch_loss.copy()
        stud_ep_loss_copy = stud_epoch_loss.copy()
        arr_pass = [tm, sm, teach_ep_loss_copy, stud_ep_loss_copy, idx]
        q.put(arr_pass)
        return  

    def server_train_func(q, net_glob, teachers, global_ep, w_org=None, base_teachers=None, server_lr=1e-3):
        device=torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        student = ServerUpdate(args=args, device=device, dataset=dataset_eval, server_dataset=dataset_eval, server_idxs=server_id,  
                            test=(dataset_test, range(len(dataset_test))), w_org=w_org, base_teachers=base_teachers, server_lr=server_lr)
      
        w_swa, w_glob, train_acc, val_acc, test_acc, loss, entropy = student.train(net_glob, teachers, args.log_dir, global_ep)
        q.put([w_swa, w_glob, train_acc, val_acc, test_acc, entropy])
        return     
     
    dist_logger = logger("DIST") 
    fedavg_logger = logger("FedAvg")  
    all_size_arr = [np.sum(cnts_dict[i]) for i in range(args.num_users)]    

    generator = None
    num_threads = args.num_threads
    tb_writer = SummaryWriter(log_dir=args.log_dir + '/')

    for iters in range(args.rounds):

        print('\nGlobal Training Round: {}'.format(iters))
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # Training
        teacher_wt = copy.deepcopy(teacher_model.state_dict())
        student_wt = copy.deepcopy(student_model.state_dict())

        
        client_teachers = [[] for i in range(args.num_users)]
        client_students = [[] for i in range(args.num_users)]

        # Training        
        for i in range(0, m, num_threads):
            processes = []
            torch.cuda.empty_cache()
            q = mp.Manager().Queue()
            num_in=0
            num_out=0

            for idx in idxs_users[i:min(m,i+num_threads)]:
                p = mp.Process(target=client_ts_train_func, args=(q, copy.deepcopy(teacher_model), copy.deepcopy(student_model), args.teach_lr, args.stud_lr, args.teach_wd, args.stud_wd, args.teach_sch_gamma, args.teach_sch_step, iters, idx, generator))
                num_in+=1
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            while num_in!=num_out:
                #running = any(p.is_alive() for p in processes)
                while not q.empty():
                    q_return = q.get()
                    client_id = q_return[-1]
                    teach_epoch_loss = q_return[2]
                    stud_epoch_loss  = q_return[3]

                    for ep in range(len(teach_epoch_loss)):
                        tb_writer.add_scalar(f'Client_{client_id}_Teacher/Train Loss' , teach_epoch_loss[ep], ep+iters*args.local_ep)
                        tb_writer.add_scalar(f'Client_{client_id}_Student/Train Loss' , stud_epoch_loss[ep], ep+iters*args.local_ep)

                    client_teachers[q_return[-1]].append(q_return[0])
                    client_students[q_return[-1]].append(q_return[1])
                    num_out+=1
                    del q_return

            for p in processes:
                p.close()

            if num_in != num_out:
                print(num_in, num_out)
                exit("Error with multi-process output. One or more outputs not received")

        size_arr = [all_size_arr[i] for i in range(args.num_users) if len(client_teachers[i])>0]
        client_teachers = [c[0] for c in client_teachers if len(c)>0]
        client_teacher_wts = [c.state_dict() for c in client_teachers]
        client_students = [c[0] for c in client_students if len(c)>0]
        client_student_wts = [c.state_dict() for c in client_students]
          
        teacher_wt_avg = FedAvg(client_teacher_wts, size_arr=size_arr)    
        teacher_model.load_state_dict(teacher_wt_avg) 

        if args.stud_ent_avg:
            print('Entropy-weighted aggregation of student models at server done!')
            student_wt_avg = FedEntropyAvg(copy.deepcopy(student_model), client_student_wts, server_data=dataset_eval ,server_id=server_id, size_arr=size_arr)    

        else:
            print('Datasize-weighted aggregation of student models at server done!')
            student_wt_avg = FedAvg(client_student_wts, size_arr=size_arr)

        student_model.load_state_dict(student_wt_avg)
        
        if iters%args.log_ep== 0:
            # log_test_net(fedavg_logger, args.acc_dir, teacher_model, tag='teacher_server', iters=iters, dataset_train=dataset_eval, dataset_test=dataset_test, train_ids=train_ids)
            log_test_net(fedavg_logger, args.acc_dir, student_model, tag='student_server', iters=iters, dataset_train=dataset_eval, dataset_test=dataset_test, train_ids=train_ids)
        
        # Generate Teachers
        if args.distill:
            distill_tlist = []

            distill_tlist.append(copy.deepcopy(teacher_model)) # Add FedAvg

            teacher_model.train()

            if args.teacher_type=="SWAG" and iters > args.warmup_ep:
                for i in range(args.num_sample_teacher):
                    base_teachers = client_teacher_wts
                    swag_model = SWAG_server(args, teacher_wt, avg_model=teacher_wt_avg, concentrate_num=1, size_arr=size_arr)
                    w_swag = swag_model.construct_models(base_teachers, mode=args.sample_teacher) 
                    teacher_model.load_state_dict(w_swag)
                    distill_tlist.append(copy.deepcopy(teacher_model))  
            else:
                base_teachers = client_teacher_wts
                print("Warming up, using DIST.")
            
            if args.use_client:
                distill_tlist+=client_teachers        
              
            # Load weights for server training
            teacher_model.load_state_dict(teacher_wt_avg)
            print("Initialize with FedAvg for server training ...")
            # update global weights
            print("Server training...")

            q = mp.Manager().Queue()
            p = mp.Process(target=server_train_func, args=(q, teacher_model,distill_tlist,iters,None, None, args.teach_server_lr))
            p.start()

            count_over=0
            p.join()

            while count_over!=1:
                while not q.empty():
                    pass_arr = q.get()
                    w_glob_mean, w_glob, ens_train_acc, ens_val_acc, ens_test_acc, entropy = pass_arr[0], pass_arr[1], pass_arr[2], pass_arr[3], pass_arr[4], pass_arr[5]
                    del pass_arr
                    count_over = count_over + 1

            if count_over != 1:
                exit("Process abrupted")

            p.close()

            del q

            teacher_model.eval()
            
            if iters%args.log_ep== 0:
                teacher_model.load_state_dict(w_glob_mean)
                # log_test_net(dist_logger, args.acc_dir, teacher_model, tag='Teach_DIST-SWA', iters=iters, dataset_train=dataset_eval, dataset_test=dataset_test, train_ids=train_ids)        

            teacher_model.load_state_dict(w_glob_mean)
            print("Sending back teacher w/ SWA!")

            distill_slist = []

            print("add FedAvg to Students list")
            distill_slist.append(copy.deepcopy(student_model)) # Add FedAvg

            student_model.train()

            if args.teacher_type=="SWAG" and iters > args.warmup_ep:
                for i in range(args.num_sample_teacher):
                    base_studs = client_student_wts
                    swag_model = SWAG_server(args, student_wt, avg_model=student_wt_avg, concentrate_num=1, size_arr=size_arr)
                    w_swag = swag_model.construct_models(base_studs, mode=args.sample_teacher) 
                    student_model.load_state_dict(w_swag)
                    distill_slist.append(copy.deepcopy(student_model))  
            else:
                base_studs = client_student_wts
                print("Warming up, using DIST.")
            
            if args.use_client:
                distill_slist+=client_students          
              
            # Load weights for server training
            student_model.load_state_dict(student_wt_avg)
            print("Initialize with FedAvg for server training ...")
            # update global weights
            print("Server training...")

            q = mp.Manager().Queue()
            p = mp.Process(target=server_train_func, args=(q,student_model,distill_slist,iters,None, None, args.stud_server_lr))
            p.start()

            count_over=0
            p.join()

            while count_over!=1:
                while not q.empty():
                    pass_arr = q.get()
                    w_glob_mean, w_glob, ens_train_acc, ens_val_acc, ens_test_acc, entropy = pass_arr[0], pass_arr[1], pass_arr[2], pass_arr[3], pass_arr[4], pass_arr[5]
                    del pass_arr
                    count_over=count_over+1

            if count_over != 1:
                exit("Process abrupted")

            p.close()

            del q

            student_model.eval()
            
            if iters%args.log_ep== 0:
                student_model.load_state_dict(w_glob_mean)
                log_test_net(dist_logger, args.acc_dir, student_model, tag='Stud_DIST-SWA', iters=iters , dataset_train=dataset_eval, dataset_test=dataset_test, train_ids=train_ids)        

            student_model.load_state_dict(w_glob_mean)
            print("Sending back student w/ SWA!")

            del client_teachers
            del client_students
    torch.save(teacher_model.state_dict(), os.path.join(args.log_dir, "teacher_model"))
    torch.save(student_model.state_dict(), os.path.join(args.log_dir, "student_model"))