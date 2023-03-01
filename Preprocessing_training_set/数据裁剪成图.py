import os
import random
import xml.etree.ElementTree as ET
import netCDF4 as nc
import numpy as np
from PIL import Image
from utils.utils import get_classes

import os
import xml.etree.cElementTree as ET




'''增强图片裁剪训练'''
# VOCdevkit_path  = 'E:\\photo\\04\\Annotations'
VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
classes_path        = 'model_data/pre_eddy.txt'

photo_nums  = np.zeros(len(VOCdevkit_sets))


files=open('D:\\PC\\yolov7-pytorch-master\\Preprocessing_training_set\\2007_train.txt','r')
file_clip=open('D:\\PC\\yolov7-pytorch-master\\Preprocessing_training_set\\2007_train_clip2.txt','w')
for file in files:
    AE=[]
    CE=[]
    BOX=[]

    for obj in file[:-1].split(' ')[1:]:
        if obj.split(',')[4] == '0':
            AE.append(list(map(int, obj.split(',')[0:4])))
        elif obj.split(',')[4] == '1':
            CE.append(list(map(int, obj.split(',')[0:4])))
        elif obj.split(',')[4] == '2':
            BOX.append(list(map(int, obj.split(',')[0:4])))
    dataset = Image.open(file.split(' ')[0])
    data=np.array(dataset)
    for num,box in enumerate(BOX):
        file_clip.write(file[:-1].split(' ')[0][:11]+'\\ImageSets\\'+file[:-1].split(' ')[0].split('\\')[-1][:-4]+str(num)+'.jpg')
        for ae in AE:
            if ae[0]>box[0] and ae[1]>box[1] and ae[2]<box[2] and ae[3]<box[3]:
                # file_clip.write(str(ae[0]-box[0]),str(ae[0]-box[0]),str(ae[0]-box[0]),str(ae[0]-box[0]))
                file_clip.write(" " + ",".join([str(c-d) for c,d in zip(ae,np.tile(np.array(box[0:2]),2))]) + ',' + str(0))
        for ce in CE:
            if ce[0] > box[0] and ce[1] > box[1] and ce[2] < box[2] and ce[3] < box[3]:
                file_clip.write(" " + ",".join([str(c-d) for c,d in zip(ce,np.tile(np.array(box[0:2]),2))]) + ',' + str(1))
        file_clip.write('\n')
        '''生成box样本图片'''

        data_clip=data[box[1]:box[3],box[0]:box[2]]

        # image = np.array(data_clip, dtype=np.uint8).transpose(1, 2, 0)
        # image = np.array(image).transpose(1, 2, 0)
        pil_image = Image.fromarray(data_clip)
        pil_image.save('E:\\photo\\00\\ImageSets\\'+file[:-1].split(' ')[0][12:-4]+str(num)+'.jpg')
        '''生成对应的xml文件'''


'''chl的NC文件 进行裁剪训练'''
# def float4color(zero2one):
#     x = zero2one/100 * 256 * 256 * 256
#     r = x % 256
#     g = ((x - r)/256 % 256)
#     b = ((x - r - g * 256)/(256*256) % 256)
#     r=r.filled(255)
#     g=g.filled(255)
#     b=b.filled(255)
#     r = r.astype('int')
#     g = g.astype('int')
#     b = b.astype('int')
#     return (r,g,b)
#
# # VOCdevkit_path  = 'E:\\photo\\04\\Annotations'
# VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
# classes_path        = 'model_data/pre_eddy.txt'
#
# photo_nums  = np.zeros(len(VOCdevkit_sets))
#
#
# files=open('D:\\PC\\yolov7-pytorch-master\\预处理训练集\\2007_train.txt','r')
# file_clip=open('D:\\PC\\yolov7-pytorch-master\\预处理训练集\\2007_train_clip2.txt','w')
# for file in files:
#     AE=[]
#     CE=[]
#     BOX=[]
#
#     for obj in file[:-1].split(' ')[1:]:
#         if obj.split(',')[4] == '0':
#             AE.append(list(map(int, obj.split(',')[0:4])))
#         elif obj.split(',')[4] == '1':
#             CE.append(list(map(int, obj.split(',')[0:4])))
#         elif obj.split(',')[4] == '2':
#             BOX.append(list(map(int, obj.split(',')[0:4])))
#     dataset = nc.Dataset(file.split(' ')[0])
#     data = dataset.variables['chl'][:]
#     for num,box in enumerate(BOX):
#         file_clip.write(file[:-1].split(' ')[0][:-3]+str(num)+'.jpg')
#         for ae in AE:
#             if ae[0]>box[0] and ae[1]>box[1] and ae[2]<box[2] and ae[3]<box[3]:
#                 # file_clip.write(str(ae[0]-box[0]),str(ae[0]-box[0]),str(ae[0]-box[0]),str(ae[0]-box[0]))
#                 file_clip.write(" " + ",".join([str(c-d) for c,d in zip(ae,np.tile(np.array(box[0:2]),2))]) + ',' + str(0))
#         for ce in CE:
#             if ce[0] > box[0] and ce[1] > box[1] and ce[2] < box[2] and ce[3] < box[3]:
#                 file_clip.write(" " + ",".join([str(c-d) for c,d in zip(ce,np.tile(np.array(box[0:2]),2))]) + ',' + str(1))
#         file_clip.write('\n')
#         '''生成box样本图片'''
#
#         data_clip=data[box[1]:box[3],box[0]:box[2]]
#         image = float4color(data_clip)
#         image = np.array(image, dtype=np.uint8).transpose(1, 2, 0)
#         # image = np.array(image).transpose(1, 2, 0)
#         pil_image = Image.fromarray(image)
#         pil_image.save('E:\\photo\\00\\ImageSets\\'+file[:-1].split(' ')[0][7:-3]+str(num)+'.jpg')
