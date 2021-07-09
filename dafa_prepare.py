# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:55:48 2021

@author: Ding1119
"""

from scipy import io
import scipy
import os
###运行时需要改变root值为BSD500所在的相应根目录
root = 'C:/Users/Ding1119/BDCN-master/path_to/bsds500/BSR/BSDS500'
PATH = os.path.join(root,'data\\groundTruth')


for sub_dir_name in ['train','test','val']:
    sub_pth = os.path.join(PATH,sub_dir_name)
    ##为生成的图片新建个文件夹保存
    save_pth = os.path.join(root,'data\\GT_convert',sub_dir_name)
    os.makedirs(save_pth,exist_ok=True)
    print('开始转换'+sub_dir_name+'文件夹中内容')
    for filename in os.listdir(sub_pth):
        # 读取mat文件中所有数据
        #mat文件里面是以字典形式存储的数据
        #包括 dict_keys(['__globals__', 'groundTruth', '__header__', '__version__'])
        #如果要用到'groundTruth']中的轮廓
        #x['groundTruth'][0][0][0][0][1]为轮廓
        #x['groundTruth'][0][0][0][0][0]为分割图
        data = io.loadmat(os.path.join(sub_pth,filename))
        edge_data = data['groundTruth'][0][0][0][0][1]
        #存储的是归一化后的数据：0<x<1
        #因此需要还原回0<x<255
        edge_data_255 = edge_data * 255
        new_img_name = filename.split('.')[0]+'.jpg'
        print(new_img_name)
        scipy.misc.imsave(os.path.join(save_pth,new_img_name), edge_data_255)  # 保存图片
