import os
import random

path = r'E:\Segmentation\Datasets\Training_data\CMOST\CUTTED\raw'
name_ls = os.listdir(path)
out0 = open(r'E:\Segmentation\Datasets\Training_data\CMOST\CUTTED\train.txt', 'w')
out1 = open(r'E:\Segmentation\Datasets\Training_data\CMOST\CUTTED\val.txt', 'w')

random.shuffle(name_ls)
for i in range(len(name_ls)):
    if i % 10 == 0:
        out1.write(name_ls[i] + '\n')
    else:
        out0.write(name_ls[i] + '\n')