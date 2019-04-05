# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:47:58 2018

@author: 60236
"""
import math
import torch
import torchvision.models as models
from torch import nn
from torchvision import models
import torch.utils.model_zoo as model_zoo
from layers import up, out, head

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out
    


class my_resnet(models.resnet.ResNet):
    
    def __init__(self,block, layers):
        models.resnet.ResNet.__init__(self, block, layers)
        self.head = head(3,64)
        
        self.up1 = up(2048,1024)
        self.up2 = up(1024,512)
        self.up3 = up(512,256)
        self.up4 = up(256,64)
        self.out = out(64,3)
        
    def forward(self, x):
        head = self.head(x)
        
        x = self.conv1(x)
        x1 = self.bn1(x)
        x2 = self.relu(x1) #? 64,64,64
        xx = self.maxpool(x2) # 64,32,32
        layers = [self.layer1,self.layer2, self.layer3, self.layer4]
        
        #------
        # down
        #------
        a = layers[0](xx) #? 256,16,16
        b = layers[1](a) #? 512,8,8
        c = layers[2](b) #? 1024, 4, 4
        d = layers[3](c) #? 2048, 2, 2
        #-------
        #  up
        #-------
        up =self.up1(d,c) # ? 1024,4,4
        up = self.up2(up,b)# 512,8
        up = self.up3(up,a)#256,16
        up = self.up4(up,x)
        
        ret = self.out(up,head)
        
        return nn.Sigmoid()(ret)
    
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = my_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
if __name__=="__main__":
    import cv2
    from PIL import Image
#    im0 = Image.open('./data/train/image/0.png')
#    im = cv2.imread('./data/train/image/0.png')
    x = torch.randn(1,3,512,512).cuda()
    n = resnet50().cuda()
    y = n(x)
    print('\n',x.shape)
    print(y.shape)