# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:19:16 2018

@author: 60236
"""

from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import glob
from PIL import Image
import random
import cv2

class test_data(Dataset):
    def __init__(self,img_dir,transform):
        self.transform = transforms.Compose(transform)
        self.files = glob.glob(img_dir+'./*.*')
    def __getitem__(self, index):
        img = cv2.imread(self.files[index % len(self.files)])
        img = self.transform(img)
        
        return img
    
    def __len__(self):
        return len(self.files)


class datagen(Dataset):
    def __init__(self,root,sr_transform=None, ta_transform=None, train=True, test_ratio=0.2):
        self.sr_transform = transforms.Compose(sr_transform)
        self.ta_transform = transforms.Compose(ta_transform)
        self.train = train
        
        ###读取保存所有的原图和标签路径。
        self.src, self.tar = [], []
        with open(root,'r') as f:
            lines = f.readlines()  
        for line in lines:
            src, tar = line.strip().split(' ')
            self.src.append(src)
            self.tar.append(tar)
        ###80%用于训练  20%用于测试
        num_test = len(self.src) * test_ratio * (-1)
        num_test = int(num_test)
        
        self.src = self.src[:num_test] if self.train else self.src[num_test:]
        self.tar = self.tar[:num_test] if self.train else self.tar[num_test:]
            
    def __getitem__(self,index):
        
        
        sr_img = cv2.imread(self.src[index % len(self.src)])
        ta_img = cv2.imread(self.tar[index % len(self.tar)])
        
        if self.train == True:
        ##################  闅忔満姘村钩缈昏浆 ###################################
        
            if random.random()<0.5:
                sr_img = cv2.flip(sr_img,0)
                ta_img = cv2.flip(ta_img,0)
        ##################  闅忔満鍨傜洿缈昏浆 ###################################
            if random.random()<0.5:
                sr_img = cv2.flip(sr_img,1)
                ta_img = cv2.flip(ta_img,1)
                
            if random.random()<0.5:
                sr_img = cv2.flip(sr_img,-1)
                ta_img = cv2.flip(ta_img,-1)
            
        sr_img = self.sr_transform(sr_img)
        ta_img = self.ta_transform(ta_img)
        
        
        
        return {'sr':sr_img, 'ta': ta_img}
    def __len__(self):
        return len(self.src)
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    root = r'./data/ls/'
    transform = [#transforms.Resize((416,416),interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                 ]
    da = datagen(root,transform,transform)
    li = DataLoader(da, shuffle=True, batch_size=1)
    print(len(li))
    
    for j,  i in enumerate(li):
       
        
        sr = i['sr'].squeeze(0).permute(1,2,0).contiguous()
        ta = i['ta'].squeeze(0).permute(1,2,0).contiguous()
        print(sr.shape)

        plt.subplot(121)
        plt.imshow(sr)
        plt.subplot(122)
        plt.imshow(ta,'gray')
        break