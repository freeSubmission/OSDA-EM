######################################################################################
#   OSDA-EM: Open Set Domain Adaptation with Entropy Minimization
#   dataset: BCIS
#   domain transferring task: source_name -->  target_name
#   Date: created on Jan. 18, 2019
######################################################################################
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import mat_data_loader as data_loader
import DecafNet as models
import numpy as np
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Training settings
t = 0.5
dim_ft = 1024
batch_size = 64
num_classes = 11
num_experiments = 10
epochs = 200
lr = 0.002
momentum = 0.9
no_cuda =False
seed = 999
log_interval = 10
l2_decay = 5e-4
root_path = "/home/xfuwu/data/dense_setup_decaf7/"
source_name = "bing"         #from: sun, imagenet, caltech256, bing
target_name = "caltech256"  #from: sun, imagenet, caltech256, bing
source = 1
target = 0
cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': False} if cuda else {}

source_loader = data_loader.load_and_partition(root_path, source_name, source, batch_size)
target_train_loader = data_loader.load_and_partition(root_path, target_name, target, batch_size)
target_test_loader = data_loader.load_and_partition(root_path, target_name, target, batch_size)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)

def label_dist_dataset():
    label_dist_src = list(np.zeros(num_classes))
    source_all_loader = data_loader.load_and_partition(root_path, source_name, source, len_source_dataset)
    data_source_iter = iter(source_all_loader)

    source_data, source_label = data_source_iter.next()

    for cls in list(source_label):
        label_dist_src[cls] += 1.0

    label_dist_tgt = list(np.zeros(num_classes))

    target_all_loader = data_loader.load_and_partition(root_path, target_name, target,len_target_dataset)
    data_target_iter = iter(target_all_loader)

    target_data, target_label = data_target_iter.next()
    for cls in list(target_label):
        label_dist_tgt[cls] += 1.0

    return (label_dist_src, label_dist_tgt)

def train(epoch, model, ws):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)

    print("learning rate：", LEARNING_RATE)

    optimizer_osda = torch.optim.SGD([
        #{'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters()},
    ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_train_loader)

    i = 1
    loss_domain = 0.
    while i <= len_source_loader:
        model.train()

        source_data, source_label = data_source_iter.next()

        ws_batch = [ws[cls] for cls in list(source_label)]
        ws_batch = torch.FloatTensor(ws_batch).cuda()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        src_output, fts = model(source_data)  #softmax forms

        ## Source classification loss
        clabel_src = F.softmax(src_output)
        label_loss = (ws_batch * F.nll_loss(clabel_src.log(), source_label, reduce=False)).mean()

        target_data, target_label = data_target_iter.next()

        if i % len_target_loader == 0:
            data_target_iter = iter(target_train_loader)
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data = Variable(target_data)
        tgt_output, fts = model(target_data)
        clabel_tgt_all = F.softmax(tgt_output)

        ## compute binary balancing loss
        pb_pred_tgt_all = clabel_tgt_all.sum(dim=0)
        pt_2cls_pred = torch.cuda.FloatTensor([0,0])
        pt_2cls_pred[0] = pb_pred_tgt_all[0:num_classes-1].sum()
        pt_2cls_pred[1] = pb_pred_tgt_all[num_classes-1]
        pt_2cls_pred = 1.0/pt_2cls_pred.sum() * pt_2cls_pred  #normalizatoin to a prob. dist.
        target_bb_loss = torch.sum((pt_2cls_pred * torch.log(pt_2cls_pred + 1e-8)))
        # weighting factor = 0.5
        target_bb_loss = 0.5 * target_bb_loss

        ## compute binary cross-entropy loss
        clabel_tgt = clabel_tgt_all[:, 0:num_classes-1]
        pp = clabel_tgt_all[:, num_classes - 1]
        target_bce_loss = (- t * torch.log(pp + 1e-8) - (1 - t) * torch.log(1.0 - pp + 1e-8)).mean()
        #target_bce_loss = (t * (math.log(t + 1e-8) - torch.log(pp + 1e-8)) + (1 - t) * (math.log(1 - t + 1e-8) - torch.log(1 - pp + 1e-8))).mean()
        # weighting factor = 0.3
        target_bce_loss = 0.3 * target_bce_loss

        ## compute category-diversity loss
        pb_pred_tgt_all = clabel_tgt_all.sum(dim=0)
        pb_pred_tgt = pb_pred_tgt_all[0:num_classes-1]
        pb_pred_tgt = 1.0/pb_pred_tgt.sum() * pb_pred_tgt  #normalizatoin to a prob. dist.
        target_div_loss =   - (- torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-8)))) #math.log(num_classes-1)
        # weighting factor = 0.5
        target_div_loss = 0.5 * target_div_loss

        target_entropy_loss= -torch.mean((clabel_tgt_all * torch.log(clabel_tgt_all + 1e-8)).sum(dim=1))
        target_entropy_loss = 1.0 * target_entropy_loss

        total_loss = label_loss + target_entropy_loss + target_div_loss + target_bce_loss + target_bb_loss

        ##1: Training shared network and label classifier: With confusion loss, it force a domain-invariant representation.
        optimizer_osda.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer_osda.step()
        optimizer_osda.zero_grad()

        i = i + 1

    print('Train Epoch:{} ||toal_Loss: {:.2f} ||entropy_Loss: {:.2f} ||label_Loss: {:.2f} ||div_Loss: {:.2f} ||bb_Loss: {:.2f}'.format(
            epoch, total_loss, target_entropy_loss.data[0], label_loss.data[0], target_div_loss.data[0],
            target_bb_loss.data[0]))

def test(model):
    model.eval()
    test_loss = 0
    test_ent_loss = 0
    correct = 0
    correct_shared = 0
    num_shared_cls = 0
    num_pred_unknown = 0

    ##correct_cls_list
    num_cls = list(np.zeros(num_classes))
    correct_cls = list(np.zeros(num_classes))
    pred_cls = list(np.zeros(num_classes))
    for target_data, target_label in target_test_loader:
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data, target_label = Variable(target_data), Variable(target_label)
        tgt_output, fts = model(target_data)  # prob
        out_tgt = F.softmax(tgt_output)

        test_loss += F.nll_loss(out_tgt.log(), target_label, size_average=False).data[0]  # sum up batch loss

        # values, indices = tensor.max(0)
        pred = out_tgt.data.max(1)[1]  # get the index of the max log-probability

        correct += pred.eq(target_label.data.view_as(pred)).cpu().sum().item()
        for cls in range(num_classes):
            cls_inds = (target_label == cls).nonzero()
            cls_pred_num = (pred == cls).nonzero().size(0)
            pred_cls[cls] += cls_pred_num

            num_samples = cls_inds.size(0)
            cls_target_label = target_label[cls_inds]
            num_cls[cls] += num_samples
            cls_pred = pred[cls_inds]
            correct_cls[cls] += cls_pred.eq(cls_target_label.data.view_as(cls_pred)).cpu().sum().item()

        ##exclude the unknown-class
        shared_cls_inds = (target_label - (num_classes - 1)).nonzero()
        inds_size = list(shared_cls_inds.size())
        num_shared_cls += inds_size[0]
        num_pred_unknown += (pred == (num_classes - 1)).sum().item()

        entropy_loss = -(out_tgt * torch.log(out_tgt + 1e-8)).sum(dim=1)
        test_ent_loss += entropy_loss.data[0]

        shared_target_label = target_label[shared_cls_inds]
        shared_pred = pred[shared_cls_inds]
        correct_shared += shared_pred.eq(shared_target_label.data.view_as(shared_pred)).cpu().sum().item()

    test_loss /= len_target_dataset
    test_ent_loss /= len_target_dataset

    print('num_pred_unknown', num_pred_unknown)
    print('num_samples_per_cls', num_cls)
    print('corrects_per_cls', correct_cls)
    print('preds_per_cls', pred_cls)

    acc_os = 0.0
    acc_ostar = 0.0
    for cls in range(num_classes - 1):
        acc_ostar += correct_cls[cls] / num_cls[cls]
    acc_os = acc_ostar + correct_cls[num_classes - 1] / num_cls[num_classes - 1]
    acc_os = acc_os / num_classes
    acc_ostar = acc_ostar / (num_classes - 1)

    print('\n{} set: Acc-OS*: {:.2f}%, Acc-OS: {:.2f}%  \n'.format(
        target_name, 100. * acc_ostar, 100. * acc_os))

    print('\n{} set: Average loss: {:.3f}, Ent Loss：{:.3f}, Acc-Shared: {:.2f}%, Acc-All: {:.2f}%  \n'.format(
        target_name, test_loss, test_ent_loss, 100. * correct_shared / num_shared_cls,
                                               100. * correct / len_target_dataset))

    return (correct_shared, correct, acc_os, acc_ostar)

if __name__ == '__main__':
    ws, wt = label_dist_dataset()
    print('label_dist_src',ws)
    print('label_dist_tgt',wt)
    wsmax = max(ws)
    wtmax = max(wt)

    target_shared_samples = sum(wt[0:len(wt) - 1])
    target_total_samples = sum(wt)
    target_unknown_samples = target_total_samples - target_shared_samples

    print('target_total_samples', target_total_samples)
    print('target_shared_samples', target_shared_samples)

    ws_inv = []
    for x in ws:
        if (x > 0):
            ws_inv.append(wsmax / x)
        else:
            ws_inv.append(0.0)
    ##update ws
    ws = ws_inv
    print('label_dist_src', ws)

    acc_os = list(np.zeros(num_experiments))
    acc_ostar = list(np.zeros(num_experiments))
    accuracy = list(np.zeros(num_experiments))
    acc_share = list(np.zeros(num_experiments))
    acc_max = list(np.zeros(num_experiments))

    for ex in range(num_experiments):
        model = models.DecafNet(num_classes=num_classes, s=16.0)
        if cuda:
            model.cuda()
        #model = load_pretrain(model)
        correct = 0
        for epoch in range(1, epochs + 1):
            train(epoch, model, ws)
            t_correct_share, t_correct, t_acc_os, t_acc_ostar = test(model)

            if t_correct > correct:
                correct = t_correct
            print('Experi-No: {} source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
                ex + 1, source_name, target_name, correct, 100. * correct / len_target_dataset))
        acc_os[ex] = t_acc_os
        acc_ostar[ex] = t_acc_ostar
        accuracy[ex] = 100. * t_correct / len_target_dataset
        acc_max[ex] = 100. * correct / len_target_dataset
        acc_share[ex] = 100. * t_correct_share / target_shared_samples

    print('Accs:', accuracy)
    print('Max-Accs:', acc_max)
    print('Acc_share:', acc_share)
    avg_acc = sum(accuracy) / len(accuracy)
    print('Avg-Acc:', avg_acc)
    avg_acc_share = sum(acc_share) / len(acc_share)
    print('Avg-Acc-share:', avg_acc_share)
    print('---------------------------------------------------------------')
    print('acc_os:', acc_os)
    print('acc_os*:', acc_ostar)
    avg_acc_os = sum(acc_os) / len(acc_os)
    print('Avg-acc-os:', avg_acc_os)
    avg_acc_ostar = sum(acc_ostar) / len(acc_ostar)
    print('Avg-acc-os*:', avg_acc_ostar)