# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:06:38 2018

@author: 60236
"""
from torch import nn
import math
import torch
import torch.nn.functional as F

class conv_Relu(nn.Module):
    def __init__(self,inplanes, outplanes, kernel=3):
        super(conv_Relu, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(inplanes,outplanes,kernel,padding=1),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(outplanes,outplanes,kernel,padding=1),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                )
    def forward(self,x):
        return self.conv(x)

class head(nn.Module):
    def __init__(self, inplanes, outplanes,kernel=3):
        super(head, self).__init__()
        self.head = nn.Sequential(
                nn.Conv2d(inplanes, outplanes,kernel, padding=1),
                nn.BatchNorm2d(outplanes),
                nn.LeakyReLU(0.2,inplace=True),
                )
    def forward(self,x):
        return self.head(x)

class add(nn.Module):
    def __init__(self, inplanes, outplanes, kernel=1):
        super(add,self).__init__()
        self.add = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Conv2d(inplanes,inplanes,kernel),
                )
        self.smooth = nn.Conv2d(inplanes,outplanes,kernel)
    def forward(self,x):
        out = self.add(x)
        out = self.add(x)
        return self.smooth(x+out)


class up(nn.Module):
    def __init__(self, inplanes, outplanes, mode='bilinear'):
        super(up,self).__init__()
        
        
        if mode == 'bilinear':
            self.up = nn.Sequential(
                    nn.Conv2d(inplanes,outplanes,kernel_size=1),
                    #nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
                    #F.interpolate(scale_factor=2,mode='bilinear',align_corners=True)
                    )
        else:
            pass
        self.conv = conv_Relu(outplanes*2,outplanes)
        self.add  = add(outplanes*2,outplanes)
    def forward(self, m, n):
        m = self.up(m)
        m = F.interpolate(m,scale_factor=2,mode='bilinear',align_corners=True)
        X =  n.size()[2] - m.size()[2]
        Y =  n.size()[3] - m.size()[3]
        m = F.pad(m,(math.ceil(X/2),math.floor(X/2),math.ceil(Y/2),math.floor(Y/2)))
        x = torch.cat([m,n], 1)
        #x = self.conv(x)
        x = self.add(x)
        return x
    
class out(nn.Module):
    def __init__(self, inplanes, outplanes, mode='bilinear'):
        super(out,self).__init__()
        
        if mode == 'bilinear':
            self.up = nn.Sequential(
                    nn.Conv2d(inplanes,inplanes,kernel_size=1),
                    #nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
                    )
        else:
            pass
        
        ###64,3
        self.end = nn.Conv2d(inplanes, outplanes,kernel_size=1)
    def forward(self,x, y):
        cat = F.interpolate(self.up(x),scale_factor=2,mode='bilinear',align_corners=True) + y
        return self.end(cat)
