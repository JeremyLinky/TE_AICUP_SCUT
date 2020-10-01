from sklearn import model_selection
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
import os

size_of_test = 0.09

"""准备数据"""
def prepare_data(namelist):
    """加载铆压机的数据集"""
    feature = []
    label = []

    """构造原始数据集（1：OK；0：NOK）"""
    for i in range(0,len(namelist)):

        Ndir = os.path.join('D:/大三下/AI CUP/铆压机导出的数据', namelist[i]);
        csv_path = Ndir  # 此处要将“\”替换为“/”
        os.chdir(os.path.dirname(csv_path))  # 用os库改变目录

        data = pd.read_csv(os.path.basename(csv_path), header=11, usecols=[6, 7, 10], #取特定列的数据
                               error_bad_lines=False)  #用os.path.basename(csv_path)获取文件名
        '''判断是不是ok的'''
        labeljudge = pd.read_csv(os.path.basename(csv_path),header=0,usecols=[1],
                               error_bad_lines=False)  #用os.path.basename(csv_path)获取文件名
        judge = labeljudge.values.tolist()  # 转换成列表

        if judge[4][0]=="OK":
            label.append(1)
        else:
            label.append(0)

        exampleData = data.values.tolist()  # 转换成列表
        np.concatenate(exampleData, axis=0)
        #print(namelist[i])

        X = list()
        for i in range(-257, -1): # 左开右闭
            X.append(exampleData[i])
        # print(np.array(X).shape)
        feature.append(X)


    feature = np.array(feature)
    label = np.array(label)
    return feature, label

"""对输入的单个样本数据进行处理"""
def process_data(file_path):
    feature = []
    Ndir = os.path.join('D:/大三下/AI CUP/铆压机导出的数据', file_path);
    csv_path = Ndir  # 此处要将“\”替换为“/”
    os.chdir(os.path.dirname(csv_path))  # 用os库改变目录

    data = pd.read_csv(os.path.basename(csv_path), header=11, usecols=[6, 7, 10], #取特定列的数据
                            error_bad_lines=False)  #用os.path.basename(csv_path)获取文件名

    exampleData = data.values.tolist()  # 转换成列表
    np.concatenate(exampleData, axis=0)

    X = list()
    for i in range(-257, -1): # 左开右闭
        X.append(exampleData[i])
        # print(np.array(X).shape)
    feature.append(X)
    return feature

"""构建训练集类"""
class train_mini_train():

    def __init__(self,feature, label):
        # 加载数据集
        self.X, self.y = \
            feature,label

        print('样本总数：', len(self.X))
        print('样本特征维度：', len(self.X[0]))

        # 数据集切分
        self.X_train, self.X_test, self.y_train, self.y_test = \
            model_selection.train_test_split(self.X, self.y, test_size=size_of_test,random_state=0)

        print('构建训练集样本总数：', len(self.y_train))

    def __len__(self):
        # 返回训练集数据量
        return len(self.y_train)

    def __getitem__(self, index):
        return torch.tensor(self.X_train[index].reshape(12, 8, 8), dtype=torch.float32), self.y_train[index]

"""构建测试集类"""
class test_mini_test():

    def __init__(self, feature, label):
        self.X, self.y = feature, label
        self.X_train, self.X_test, self.y_train, self.y_test = \
            model_selection.train_test_split(self.X, self.y, test_size=size_of_test, random_state=0)

        print('构建测试集样本总数：', len(self.y_test))

    def __getitem__(self, index):
        return torch.tensor(self.X_test[index].reshape(12, 8, 8), dtype=torch.float32), self.y_test[index]

    def __len__(self):
        return len(self.y_test)

"""定义神经网络结构类"""
class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(  # (12, 8, 8)
            nn.Conv2d(in_channels=12, out_channels=4, kernel_size=2, stride=1, padding=1),  # (4, 8, 8)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(),  # (4, 8, 8)
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)  # (4,4,4) 不改变通道数
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=1, padding=1),  # (8,5,5)
        )
        self.conv5 = nn.Sequential(
            nn.ReLU(),  # (8,5,5)
        )
        self.conv6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)  # (8,2,2)
        )
        self.fc = nn.Linear(8 * 2 * 2, 2)  # (10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1)  # 相当于Flatten
        x = self.fc(x)
        return x

"""验证时计算精度"""
def eval_on_dataloader(name, loader, len, net):
    acc = 0.0
    with torch.no_grad():
        for data in loader:
            datas, labels = data
            outputs = net(datas)
            # torch.max返回两个数值，一个[0]是最大值，一个[1]是最大值的下标
            predict_y = torch.max(outputs, dim=1)[1]
            #print(predict_y)
            acc += (predict_y == labels.long()).sum().item()
        accurate = acc / len
        return accurate

"""测试时计算精度"""
def eval_on_output(labels, outputs):
    acc = 0.0
    predict = torch.max(outputs, dim=1)[1]
    #print(predict)
    acc += (predict == labels.long()).sum().item()
    accurate = acc / len(labels)
    return accurate