from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
import time

from utils.return_dataset_mme import return_dataset
#return_dataset_balance_self_hard_fast, return_dataset_balance_self_fast

from utils.loss import entropy, adentropy
from utils.loss import PrototypeLoss, CrossEntropyKLD
from utils.loss2 import PrototypeLoss2

from pdb import set_trace as breakpoint

# from log_utils.utils import ReDirectSTD


# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--protow',type=float, default=0.95)
parser.add_argument('--cudanum', type=str, default='2',help='resume from checkpoint')
parser.add_argument('--ways', type=int, default=10, help='number of classes sampled')
parser.add_argument('--alpha', type=float, default=0.1, help='loss weight')
parser.add_argument('--beta', type=float, default=0.9, help='loss weight')
parser.add_argument('--threshold', type=float, default=0.95, help='loss weight')
parser.add_argument('--save_path', type=str, default='./MMEpa/exp4',
                    help='dir to save model')

parser.add_argument('--steps', type=int, default=100000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--gamma', type=float, default=0.4, metavar='LAM',
                    help='value of gamma')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')

# parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
#                     help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10000, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=10000, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='alexnet',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home', 'visda'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--src_shots', type=int, default=10, help='number of samples per source classes')
parser.add_argument('--trg_shots', type=int, default=3, help='number of samples per target classes')
parser.add_argument('--log_file', type=str, default='./temp.log',
                    help='dir to save checkpoint')
parser.add_argument('--align_type', type=str, default='proto',
                    choices=['proto'],
                    help='alignment type')
parser.add_argument('--kld', action='store_true', default=False,
                    help='use kld')
parser.add_argument('--w_kld', type=float, default=0.1, help='loss weight')
parser.add_argument('--labeled_hard', action='store_true', default=False,
                    help='apply hard transform on labeled data')
# parser.add_argument('--label_smooth', type=bool, default=False, help='loss weight')
parser.add_argument('--label_smooth', action='store_true', default=False,
                    help='use label smooth')
parser.add_argument('--resume', action='store_true', default=False,
help='resume from checkpoint')


args = parser.parse_args()
strd= "cuda:"+args.cudanum
device=strd if torch.cuda.is_available() else "cpu"
print(device)
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))

print('Dataset:%s\tSource:%s\tTarget:%s\tLabeled num perclass:%s\tNetwork:%s\t' % (
args.dataset, args.source, args.target, args.num, args.net))
# record_dir = os.path.join(args.save_path,args.cudanum)
record_dir = os.path.join(args.save_path,'%s_%s_%s' % (args.dataset, args.method, args.net))
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
t = time.localtime()
record_train_file = os.path.join(record_dir, '%s_to_%s_time_%d_%d_%d' % (
args.source, args.target, t.tm_mon, t.tm_mday, t.tm_hour))
record_val_file = os.path.join(record_dir, '%s_to_%s_time_%d_%d_%d' % (
args.source, args.target, t.tm_mon, t.tm_mday, t.tm_hour))

with open(record_train_file,'a') as f:
    f.write(str(args))

source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset(args)



torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    G.load_state_dict(torch.load('alexnet-owt-4df8aa71.pth'),strict=False)
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

protosize = torch.zeros(len(class_list),inc)
prototype = torch.FloatTensor(1).to(device)
prototype = Variable(prototype)
prototype.resize_(len(class_list),inc).copy_(protosize)


params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list), inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)

weights_init(F1)
lr = args.lr
G.to(device)
F1.to(device)

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
im_data_tu2 = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)
aug_labels = torch.LongTensor(1)

im_data_s = im_data_s.to(device)
im_data_t = im_data_t.to(device)
im_data_tu = im_data_tu.to(device)
im_data_tu2 = im_data_tu2.to(device)
gt_labels_s = gt_labels_s.to(device)
gt_labels_t = gt_labels_t.to(device)
sample_labels_t = sample_labels_t.to(device)
sample_labels_s = sample_labels_s.to(device)
aug_labels = aug_labels.to(device)

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
im_data_tu2 = Variable(im_data_tu2)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)
aug_labels = Variable(aug_labels)

# if os.path.exists(args.checkpath) == False:
#     os.mkdir(args.checkpath)

def train():

    best_acc_test = 0.0
    best_acc_val = 0.0

    G.train()
    F1.train()
    # head.train()
    #D.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    # optimizer_h = optim.SGD(list(head.parameters()), lr=1.0, momentum=0.9,
    #                         weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        # optimizer_h.zero_grad()
        
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().to(device)
    if args.kld:
        criterion_un = CrossEntropyKLD(num_class=len(class_list), mr_weight_kld=args.w_kld)
    if args.align_type=='proto':
        criterion_aux = PrototypeLoss2(ways=args.ways, trg_shots=args.trg_shots, src_shots=args.src_shots)    
    else:        
        raise ValueError('alignment method cannot be recognized.')

    
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)

    # breakpoint()
    counter = 0

    # is_train_dsne = False

    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_step = checkpoint['step']
            best_acc_test = checkpoint['best_acc_test']
            # prototype = checkpoint['prototype']
            G.load_state_dict(checkpoint['state_dict_G'])            
            optimizer_g.load_state_dict(checkpoint['optimizer_G'])
            F1.load_state_dict(checkpoint['state_dict_F'])            
            optimizer_f.load_state_dict(checkpoint['optimizer_f'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    for step in range(start_step, all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)  
        lr = optimizer_f.param_groups[0]['lr']
        
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
        im_data_tu2.resize_(data_t_unl[1].size()).copy_(data_t_unl[1])
        zero_grad_all()

        data = torch.cat((im_data_s, im_data_t, im_data_tu), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        
        ###################################
        output = G(data)
        out1 = F1(output)
        ns = im_data_s.size(0)
        nt = im_data_t.size(0)
        nl = ns + nt
        nu = im_data_tu.size(0)

        loss_c = criterion(out1[:nl], target)     

        pseudo_label = torch.softmax(out1[nl:nl+nu].detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        # if args.kld:
        #     loss_u = criterion_un(out1[nl+nu:], targets_u, mask)
        # else:            
        #     loss_u = (F.cross_entropy(out1[nl+nu:], targets_u, reduction='none') * mask).mean()


        if args.alpha == 0:
            proto_loss = 0
        else:
            proto_loss,nwp = criterion_aux(args.protow,device,step,ns,nt,nu,prototype,output,target,max_probs,mask, targets_u)
            prototype.copy_(nwp)
            # proto_loss = criterion_aux(gt_labels_s, output[:ns], output[ns:ns+nt], normalize_feature=True)

        loss_comb = loss_c + args.alpha * proto_loss

        loss_comb.backward()
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()

        # unl_cat = torch.cat((im_data_tu,im_data_tu2),0)
        unlabel_raw_output = G(im_data_tu)
        loss_t = adentropy(F1, unlabel_raw_output, args.lamda)                                         
        loss_t.backward()
        optimizer_f.step()
        optimizer_g.step()

        if step % args.log_interval == 0 and step > 0:
            log_train = 'Ep: {} lr: {}, loss_all: {:.6f}, loss_c: {:.6f}, loss_d: {:.6f}, loss_mme: {:.6f}'.format(step, lr, \
                    loss_comb, loss_c, proto_loss, -loss_t)
            print(log_train)
            with open(record_train_file,'a') as f:
                f.write(log_train+'\n')

        if step % args.save_interval == 0 and step > 0:
            loss_test, acc_test = test(target_loader_test)
            loss_val, acc_val = test(target_loader_val)
            G.train()
            F1.train()

            is_train_dsne = True

            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                best_acc_test = acc_test
                print('saving model')
                model_path=os.path.join(args.save_path,"save_model")
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                filename = os.path.join(model_path, "{}_{}_to_{}_step_{}_{}.pth".
                                        format(args.log_file, args.source, args.target, step, args.net))
                state = {'step': step + 1,
                    'state_dict_G': G.state_dict(),
                    'optimizer_G' : optimizer_g.state_dict(),
                    'state_dict_F': F1.state_dict(),                
                    'optimizer_f' : optimizer_f.state_dict(),
                    'best_acc_test' : best_acc_test,
                    'prototype' : prototype
                    }
                torch.save(state, filename)

            # print('best acc test %f' % (best_acc_test))
            log_val = 'Best_acc_test:{} loss_val:{} acc_val:{} loss_test:{} acc_test:{}\n'.format(best_acc_test, loss_val, acc_val, loss_test, acc_test)
            print(log_val)
            with open(record_train_file, 'a') as f:
                f.write(log_val)
           
            G.train()
            F1.train()
      


def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().to(device)
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.4f}%)\n'.
          format(test_loss, correct, size,
                 100. * float(correct) / size))
    return test_loss.data, 100. * float(correct) / size


train()
