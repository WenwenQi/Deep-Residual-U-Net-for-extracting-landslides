# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:43:14 2018

@author: 60236
"""

from model import resnet50
from datasets import datagen

import torch
import torch.nn as nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import eval_model
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of progress')
parser.add_argument('--batch_size', type=int, default=5, help='size of each image batch')
parser.add_argument('--img_size', type=int, default=600, help='size of each image dimension')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--root', type=str, default=r'./data/ls/shuffle_train_test_1150.txt', help='path to images file')
args = parser.parse_args()
print(args)

os.makedirs('./weights',exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = resnet50().cuda()

if args.resume:
    print('\n==> Resuming from checkpoint..')
    checkpoint = torch.load('./weights/ckpt.pth')
    net.load_state_dict(checkpoint['weights'])
    best_loss = checkpoint['cur_loss']
    start_epoch = checkpoint['epoch']
else:
    print('\ninit weights from pretrained model ...')
    net.load_state_dict(torch.load('./weights/init_kaim_uni_weights.pth')) # init_weights.pth
    best_loss = float('inf')
    start_epoch = 0
#para_optim = []
#for i, k in enumerate(net.children()):
#    
#    if i <= 9:
#        for param in k.parameters():
#            param.requires_grad = False
#    else:
#        for param in k.parameters():
#            para_optim.append(param)

transform = [  # transforms.Resize((args.img_size,args.img_size),interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ]

dataset = datagen(args.root,transform,transform, train=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

test  = datagen(args.root,transform,transform, train=False)
test_loader = DataLoader(test, batch_size=3)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)
scheduler = lr_scheduler.StepLR(optimizer,20,0.1)  #ajust lr  every 20 epochs to 10%

criterion = nn.BCELoss(reduction='elementwise_mean').to(device)


print('\ntrain samples is: ',len(dataset))
print(' eval samples is: ',len(test))


def train(epoch):
    print('\ntrain....')
    net.train()
    for i, img in enumerate(dataloader):
        
        optimizer.zero_grad()
        sr = img['sr'].to(device)
        label = img['ta'].to(device)
        
        out = net(sr)
        
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        
        print('[Epoch %d/%d, Bacth %d/%d --- Loss:%f ]'%(epoch, args.epochs, i, len(dataloader), loss.item()))
        
def test(epoch):
    ##  test and save
    print('\neval ....')
    net.eval()
    total_loss = 0.
    for j, img in enumerate(test_loader):
        sr = img['sr'].to(device)
        label = img['ta'].to(device)
        out = net(sr)
        loss = criterion(out, label)
        acc, pre,recall = eval_model(out, label,j)
        print('[Epoch %d/%d, Bacth %d/%d ---Loss:%.5f ]***[acc:%.2f, pre:%.2f, recall:%.2f ]'%(epoch, args.epochs, j, len(test_loader), loss.item(), acc,pre,recall))
        total_loss += loss.item()
    global best_loss
    if total_loss < best_loss:
        ckpt = {
          'weights':net.state_dict(),
          'epoch':epoch,
          'cur_loss':total_loss,
        
        }
        best_loss = total_loss
        torch.save(ckpt,'./weights/ckpt_ls_20190227.pth')
        print('model saving ......  best_loss is ', best_loss)

for epoch in range(start_epoch, start_epoch+args.epochs):
    ###   train
    train(epoch)
    test(epoch)
    scheduler.step()
    
    
    
