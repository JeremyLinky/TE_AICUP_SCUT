# 测试使用
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import os
import numpy as np

class Data(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __getitem__(self, index):
        return self.feature[index], self.label[index], ...

    def __len__(self):
        return len(self.feature)

namelist = os.listdir('D:/大三下/AI CUP/铆压机导出的数据') #数据目录

# test
feature = []
label = []

for i in range(0,len(namelist)):

    Ndir = os.path.join('D:/大三下/AI CUP/铆压机导出的数据', namelist[i]);  # 保存可视化后的折线图
    csv_path = Ndir  # 此处要将“\”替换为“/”
    os.chdir(os.path.dirname(csv_path))  # 用os库改变目录

    data = pd.read_csv(os.path.basename(csv_path), header=11, usecols=[6, 7, 10], #取特定列的数据
                           error_bad_lines=False)  #用os.path.basename(csv_path)获取文件名
    '''判断是不是ok的'''
    labeljudge = pd.read_csv(os.path.basename(csv_path),header=0,usecols=[1],
                           error_bad_lines=False)  #用os.path.basename(csv_path)获取文件名
    judge = labeljudge.values.tolist()  # 转换成列表
    #print(judge[4][0])

    if judge[4][0]=="OK":
        label.append(1)
    else:
        label.append(0)

    exampleData = data.values.tolist()  # 转换成列表
    np.concatenate(exampleData, axis=0)
    #print(namelist[i])

    X = list()
    for i in range(0, 300):
        X.append(exampleData[i])
    feature.append(X)

print(np.array(feature).shape)
print(np.array(label).reshape(42,1).shape)

dataset = Data(feature, label)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
