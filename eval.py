from __future__ import print_function


import argparse
import os
import torch
from model.resnet import resnet34, resnet50
from torch.autograd import Variable
# from tqdm import tqdm
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.return_dataset import return_dataset_test
device="cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--step', type=int, default=1000, metavar='step',
                    help='loading step')
# parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
#                     help='dir to save checkpoint')

parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--resume',default='/home/chuqs/code/jmq/pacl/pacl_lk/pacl/save_april/exp3/save_model/temp.log_real_to_sketch_step_60000_resnet34.pth'
)

# parser.add_argument('--method', type=str, default='MME',
#                     choices=['S+T', 'ENT', 'MME'],
#                     help='MME is proposed method, ENT is entropy minimization,'
#                          'S+T is training only on labeled examples')
parser.add_argument('--output', type=str, default='./output.txt',
                    help='path to store result file')
parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='which network ')
parser.add_argument('--source', type=str, default='real', metavar='B',
                    help='board dir')
parser.add_argument('--target', type=str, default='sketch', metavar='B',
                    help='board dir')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home', 'visda'],
                    help='the name of dataset, multi is large scale dataset')
args = parser.parse_args()

# print('dataset %s source %s target %s network %s' %
#       (args.dataset, args.source, args.target, args.net))

target_loader_unl, class_list = return_dataset_test(args)
# use_gpu = torch.cuda.is_available()

if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == 'resnet50':
    G = resnet50()
    inc = 2048
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
G.to(device)
F1.to(device)


# G.load_state_dict(torch.load(os.path.join(args.checkpath, "G_{}_to_{}_{}.pth.tar".format(args.source, args.target, args.net))))
# F1.load_state_dict(torch.load(os.path.join(args.checkpath, "F1_{}_to_{}_{}.pth.tar".format(args.source, args.target, args.net))))


if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        
        checkpoint = torch.load(args.resume)
        start_step = checkpoint['step']
        best_acc_test = checkpoint['best_acc_test']
        print(best_acc_test)

        G.load_state_dict(checkpoint['state_dict_G'])            
        # optimizer_g.load_state_dict(checkpoint['optimizer_G'])

        F1.load_state_dict(checkpoint['state_dict_F'])            
        # optimizer_f.load_state_dict(checkpoint['optimizer_f'])

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))





im_data_t = torch.FloatTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_t = im_data_t.to(device)
gt_labels_t = gt_labels_t.to(device)

im_data_t = Variable(im_data_t)
gt_labels_t = Variable(gt_labels_t)
# if os.path.exists(args.checkpath) == False:
#     os.mkdir(args.checkpath)


def eval(loader, output_file="output.txt"):
    G.eval()
    F1.eval()
    size = 0
    correct = 0
    # with open(output_file, "w") as f:
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            # paths = data_t[2]
            feat = G(im_data_t)
            output1 = F1(feat)
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
                # for i, path in enumerate(paths):
                #     f.write("%s %d\n" % (path, pred1[i]))

            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            # test_loss += criterion(output1, gt_labels_t) / len(loader)

    print('\n Accuracy: {}/{} F1 ({:.4f}%)\n'.format(correct, size,
                 100. * float(correct) / size))

    # return test_loss.data, 100. * float(correct) / size

eval(target_loader_unl)
