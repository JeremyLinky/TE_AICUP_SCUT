# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import numpy as np

"""可视化折线图"""
def visualize(data,name):

    exampleData = data.values.tolist() #转换成列表
    length_zu = len(exampleData)  # 得到数据行数
    length_yuan = len(exampleData[0])  # 得到每行长度

    weiyi = list()
    yali = list()
    sudu = list()

    for i in range(1, length_zu):  # 从第二行开始读取
        weiyi.append(exampleData[i][1])  # 将第一列数据从第二行读取到最后一行赋给列表weiyi
        yali.append(exampleData[i][2])  # 将第二列数据从第二行读取到最后一行赋给列表yali
        sudu.append(exampleData[i][4])  # 将第四列数据从第二行读取到最后一行赋给列表sudu


    plt.rcParams['font.sans-serif'] = ['SimHei'] #设置中文字体显示

    fig,ax1 =plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(weiyi, yali, '*', label='压力',color='orange')  # 绘制位移与压力的折线图
    ax1.set_ylabel('压力')

    ax2.plot(weiyi, sudu, '--', label='速度',color='green')  # 绘制位移与速度的折线图
    ax2.set_ylabel('速度')


    fig.legend(loc="center", bbox_transform=ax1.transAxes) #图例的摆放位置

    ax1.set_xlabel('位移')
    plt.title('铆压数据')
    plt.grid(True, linestyle = "--",color = "gray", linewidth = "0.5",axis = 'both')

    dir = os.path.join('D:\大三下\AI CUP\铆压机可视化数据', name + ".png");  # 保存可视化后的折线图
    plt.savefig(dir)

    plt.show()  # 显示折线图

"""绘制损失函数曲线"""
def plot_loss_result(losses):
    epoches = np.arange(1, len(losses) + 1, dtype=np.int32)
    plt.plot(epoches, losses, label='loss')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.legend()

"""可视化精度结果"""
def plot_train_and_test_result(train_accs,test_accs):
    epoches = np.arange(1,len(train_accs)+1,dtype=np.int32)
    plt.plot(epoches,train_accs,label='train accuracy')
    plt.plot(epoches,test_accs, label='test accuracy')
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.legend()

"""输出神经网络结构的信息"""
def printNetInfo(net):
    for name,parameters in net.named_parameters():
        print(name,":",parameters.size())