# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:48:36 2018

@author: 60236
"""

from model import resnet50
from torchvision.utils import save_image
import torch
from datasets import test_data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os


os.makedirs('./output', exist_ok=True)

#Image_path = './data/test_building/'
Image_path = '/home/qiww/data/ls/ls_validation/'
#Image_path = '/home/share/datasets/SpaceNet/AOI_4_Shanghai_Buildings_Test_public/RGB-PanSharpen/'

transform = [  # transforms.Resize((128,128),interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ]

test  = test_data(Image_path,transform)
test_loader = DataLoader(test)#,shuffle=True)


#net = resnet50()
#net.load_state_dict(torch.load('./weights/ckpt_ls_190221.pth'))
#print('loading weights success...')
#net.eval()

net = resnet50()
checkpoint = torch.load('./weights/ckpt_ls_190221.pth')
net.load_state_dict(checkpoint['weights'])
print('loading weights success...')
net = net.cuda()
net.eval()



# get image file's name
for (root, dirs, files) in os.walk(Image_path):
    image_shotname = []
    for i, img in enumerate(files):
        (img_shotname, img_extension) = os.path.splitext(img)
        if img_extension == '.tif':
            image_shotname.append(img_shotname)
        else: continue
        
for j, img in enumerate(test_loader):
    
    input_im = img
    result = net(input_im)
    
#    cat = torch.cat((input_im,result),0)
    cat = result
    save_image(cat,'./output/result_{}.png'.format(image_shotname[j]))
    print('\nsaving image ...')
#    break


