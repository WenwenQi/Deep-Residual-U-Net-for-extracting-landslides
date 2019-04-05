# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:26:39 2018

@author: 60236
"""

import random


def shuffle_txt(txt_path):
    with open(txt_path,'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            with open('shuffle_txt.txt','a') as f:
                f.write(line)
                

if __name__=='__main__':
		
		## input txt  path
		
		txt_path = r'/home/share/datasets/SpaceNet/AOI_4_Shanghai_Buildings_Train/geojson/image_and_label_list.txt'
		shuffle_txt(txt_path)
		
