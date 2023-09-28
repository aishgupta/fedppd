# =============================
# 
# main file to run below methods on datasets femnist:
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
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms

from lib.dataset_sampler import *
from lib.cnn_models import CNN_FEMNIST, CNN_FEMNIST_Stud
from trainers.Fed_Aggr import FedAvg, FedEntropyAvg
from trainers.ClientServUpdate import LocalTSUpdate, ServerUpdate
from trainers.swag import SWAG_server
from lib.options import args_parser
from lib.utils import *

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # parse args
    args = args_parser()
    if args.dataset!='femnist':
        exit('Script only works on femnist dataset')

    args.log_dir = os.path.join(args.log_dir)   

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)  

    args.acc_dir = os.path.join(args.log_dir, "acc")
    if not os.path.exists(args.acc_dir):
        os.makedirs(args.acc_dir)  
        
    model_dir = os.path.join(args.log_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    

    transform_mnist = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])

    #considering only characters from alphabets
    args.num_classes=52
    data_directory = args.femnist_data_dir

    train_data_dir = os.path.join(data_directory, "train")
    test_data_dir = os.path.join(data_directory,"test")
    server_dir = os.path.join(data_directory, "server_data")

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    su, sg, server_data = read_dir(server_dir)

    # print(users)
    args.num_users = len(users)

    with open(os.path.join(args.log_dir, "args.txt"), "w") as f:
        for arg in vars(args):
            print (arg, getattr(args, arg), file=f)

    train_dataset = {}
    test_dataset_x = []
    test_dataset_y = []
    server_dataset_x = []
    server_dataset_y = []
    train_data_all_x = []
    train_data_all_y = []

    for user in users:
        feat = np.array(train_data[user]['x'])
        feat = np.reshape(feat, (feat.shape[0],1,28,28))
        incl_indices = []

        for i, lab in enumerate(train_data[user]['y']):
            if lab>=10:
                incl_indices.append(i)

        if args.subsample != -1:
            incl_indices = subsample_data(train_data[user]['y'],incl_indices,args.subsample)

        new_feat = feat[incl_indices]
        new_lab = [(train_data[user]['y'][i] - 10) for i in incl_indices]
        train_data_all_x += [train_data[user]['x'][i] for i in incl_indices]
        train_data_all_y += [(train_data[user]['y'][i]-10) for i in incl_indices]
        t_x = torch.Tensor(new_feat)
        t_y = torch.Tensor(new_lab)
        t_y = t_y.type(torch.LongTensor)
        train_dataset[user] = TensorDataset(t_x,t_y)

        test_dataset_x += [test_data[user]['x'][i] for i in range(len(test_data[user]['x'])) if test_data[user]['y'][i]>=10]
        test_dataset_y += [(test_data[user]['y'][i] - 10) for i in range(len(test_data[user]['y'])) if test_data[user]['y'][i]>=10]


    test_dataset_x = np.array(test_dataset_x)
    test_dataset_x = np.reshape(test_dataset_x, (test_dataset_x.shape[0],1,28,28))
    test_x = torch.Tensor(test_dataset_x)
    test_y = torch.Tensor(test_dataset_y)
    test_y = test_y.type(torch.LongTensor)
    test_dataset = TensorDataset(test_x,test_y)

    train_data_all_x = np.array(train_data_all_x)
    train_data_all_x = np.reshape(train_data_all_x, (train_data_all_x.shape[0],1,28,28))
    train_data_all_x_tensor = torch.Tensor(train_data_all_x)
    train_data_all_y_tensor = torch.Tensor(train_data_all_y)
    train_data_all_y_tensor = train_data_all_y_tensor.type(torch.LongTensor)
    train_dataset_all = TensorDataset(train_data_all_x_tensor,train_data_all_y_tensor)

    for user in su:
        for i in range(len(server_data[user]['y'])):
            if server_data[user]['y'][i]>=10:
                server_dataset_x.append(server_data[user]['x'][i])
                server_dataset_y.append(server_data[user]['y'][i] - 10)

    server_dataset_x = np.array(server_dataset_x)
    server_dataset_x = np.reshape(server_dataset_x, (server_dataset_x.shape[0],1,28,28))
    server_x = torch.Tensor(server_dataset_x)
    server_y = torch.Tensor(server_dataset_y)
    server_y = server_y.type(torch.LongTensor)
    server_dataset = TensorDataset(server_x,server_y)
    
    # print(test_dataset)

    teacher_model = CNN_FEMNIST()
    student_model = CNN_FEMNIST_Stud()

    print('\nGlobal Teacher Model', teacher_model)
    print('\nGlobal Student Model', student_model)

    teacher_wt = teacher_model.state_dict()
    student_wt = student_model.state_dict()

    teacher_model.apply(weights_init) 
    student_model.apply(weights_init)

    def client_ts_train_func(q, tnet, snet, t_lr, s_lr, t_wd, s_wd, t_gamma, t_ss, iters, idx, user_seq_idx, generator=None):
        device=torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        local_eps = args.local_ep

        local = LocalTSUpdate(args=args, device=device, dataset=train_dataset[idx], idxs=None, test=(test_dataset, range(len(test_dataset))), num_per_cls=None)

        teach_model, stud_model, teach_epoch_loss, stud_epoch_loss = local.train(tnet=tnet.to(device), snet=snet.to(device), local_eps=local_eps, t_lr=t_lr, t_wd=t_wd, s_lr=s_lr, s_wd= s_wd, gamma=t_gamma, stepsiz=t_ss)
        
        tm = CNN_FEMNIST()
        sm = CNN_FEMNIST_Stud()
        tm.load_state_dict(teach_model.state_dict())
        sm.load_state_dict(stud_model.state_dict())
        
        teach_ep_loss_copy = teach_epoch_loss.copy()
        stud_ep_loss_copy = stud_epoch_loss.copy()
        
        arr_pass = [tm, sm, teach_ep_loss_copy, stud_ep_loss_copy, user_seq_idx]
        q.put(arr_pass)
        return

    def server_train_func(q, net_glob, teachers, global_ep, w_org=None, base_teachers=None, server_lr=1e-3):
        device=torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        student = ServerUpdate(args=args, device=device, dataset=server_dataset, server_dataset=server_dataset, server_idxs=range(len(server_dataset)), test=(test_dataset, range(len(test_dataset))), w_org=w_org, base_teachers=base_teachers, server_lr=server_lr)
      
        w_swa, w_glob, train_acc, val_acc, test_acc, loss, entropy = student.train(net_glob, teachers, args.log_dir, global_ep)
        q.put([w_swa, w_glob, train_acc, val_acc, test_acc, entropy])
        return

    dist_logger = logger("DIST")
    fedavg_logger = logger("FedAvg")  
    num_users = len(users)

    user_seq = {}
    all_size_arr = [0 for i in range(num_users)]
    for i in range(len(users)):
        user_seq[users[i]] = i
        all_size_arr[i] = train_dataset[users[i]].__len__()  

    generator = None
    num_threads = args.num_threads
    tb_writer = SummaryWriter(log_dir=args.log_dir + '/')  
    local_t_lr = args.teach_lr
    local_s_lr = args.stud_lr
    server_t_lr = args.teach_server_lr
    server_s_lr = args.stud_server_lr

    gamma = 0.99
    if args.distill:
        s_gamma = 0.995
        serv_gamma = 0.995
    else:
        s_gamma=0.99

    for iters in range(args.rounds):

        print('\nGlobal Training Round: ', str(iters))
        m = max(int(args.frac * num_users), 1)
        idxs_users = np.random.choice(users, m, replace=False)
        selected_seq = [user_seq[ind] for ind in idxs_users]

        #v3 Training
        teacher_wt = copy.deepcopy(teacher_model.state_dict())
        student_wt = copy.deepcopy(student_model.state_dict())

        
        client_teachers = [[] for i in range(args.num_users)]
        client_students = [[] for i in range(args.num_users)]

        #v3 Training        
        for i in range(0, m, num_threads):
            processes = []
            torch.cuda.empty_cache()
            q = mp.Manager().Queue()
            num_in=0
            num_out=0

            for idx in idxs_users[i:min(m,i+num_threads)]:
                p = mp.Process(target=client_ts_train_func, args=(q, copy.deepcopy(teacher_model), copy.deepcopy(student_model), local_t_lr, local_s_lr, args.teach_wd, args.stud_wd, args.teach_sch_gamma, args.teach_sch_step, iters, idx, user_seq[idx], generator))
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
            print('Entropy-weighted aggregation of student models')
            student_wt_avg = FedEntropyAvg(copy.deepcopy(student_model), client_student_wts, server_data=server_dataset ,server_id=list(range(len(server_dataset))), size_arr=size_arr)    

        else:
            print('Datasize-weighted aggregation of student models')
            student_wt_avg = FedAvg(client_student_wts, size_arr=size_arr)

        student_model.load_state_dict(student_wt_avg)

        local_t_lr *= gamma
        local_s_lr *= s_gamma
        
        if iters%args.log_ep== 0:
            # log_test_net(fedavg_logger, args.acc_dir, teacher_model, tag='teacher_server', iters=iters, dataset_train=train_dataset_all, dataset_test=test_dataset, train_ids=range(len(train_dataset_all)))
            log_test_net(fedavg_logger, args.acc_dir, student_model, tag='student_server', iters=iters, dataset_train=train_dataset_all, dataset_test=test_dataset, train_ids=range(len(train_dataset_all)))

        # Generate Teachers
        if args.distill:
            distill_tlist = []

            distill_tlist.append(copy.deepcopy(teacher_model)) # Add FedAvg
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
            p = mp.Process(target=server_train_func, args=(q, teacher_model,distill_tlist,iters,None, None, server_t_lr))
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
                # log_test_net(dist_logger, args.acc_dir, teacher_model, tag='Teach_DIST-SWA', iters=iters, dataset_train=train_dataset_all, dataset_test=test_dataset, train_ids=range(len(train_dataset_all)), server_id=None)
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
            p = mp.Process(target=server_train_func, args=(q, student_model,distill_slist,iters,None, None, server_s_lr))
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
            if iters%args.log_ep== 0:
                student_model.load_state_dict(w_glob_mean)
                log_test_net(dist_logger, args.acc_dir, student_model, tag='Stud_DIST-SWA', iters=iters, dataset_train=train_dataset_all, dataset_test=test_dataset, train_ids=range(len(train_dataset_all)))

            student_model.load_state_dict(w_glob_mean)
            print("Sending back student w/ SWA!")

            del client_teachers
            del client_students
            local_t_lr *= gamma
            local_s_lr *= s_gamma
            server_t_lr *= serv_gamma
            server_s_lr *= serv_gamma
    
    torch.save(teacher_model.state_dict(), os.path.join(args.log_dir, "teacher_model"))
    torch.save(student_model.state_dict(), os.path.join(args.log_dir, "student_model"))