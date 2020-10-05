import torch.utils.data as Data   #分批次loader数据集使用
import torch.nn as nn    #神经网络API
import torch.optim as opt     #神经网络优化器
import matplotlib.pyplot as plt  #数据可视化
import torch
from Visual_utils import plot_loss_result, plot_train_and_test_result, printNetInfo
import os
import utils
pwd = os.getcwd()  # 当前目录
def main():
    namelist = os.listdir(pwd+'/data') #数据目录


    """构造数据集"""
    feature, label = utils.prepare_data(namelist)

    """训练参数"""
    batch_size = 6
    learning_rate = 3e-4
    epoches = 3000

    """载入训练集与测试集数据"""
    train_data = utils.train_mini_train(feature, label)
    test_data = utils.test_mini_test(feature, label)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    """载入网络结构"""
    net = utils.SimpleNet()

    """声明损失函数及优化器"""
    loss_fn = nn.CrossEntropyLoss() #交叉熵
    optim = opt.Adam(params=net.parameters(), lr=learning_rate)

    """训练集精度、测试集精度、损失"""
    train_accs = []
    test_accs = []
    losses = []

    """训练"""
    for epoch in range(epoches):
        if epoch == 0.4*epoches:
            optim = opt.Adam(params=net.parameters(), lr=1*learning_rate)
        elif epoch == 0.8*epoches:
            optim = opt.Adam(params=net.parameters(), lr=1 * learning_rate)
        net.train()  # 训练
        for step, data in enumerate(train_loader, start=0):
            images, labels = data

            optim.zero_grad()  # 优化器梯度清0
            logits = net(images)  # 输入images 经过网络推断输出结果

            loss = loss_fn(logits, labels.long())  # 计算损失函数
            loss.backward()  # 反向传播求梯度
            optim.step()  # 优化器进一步优化

            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\repoch:%s train loss:%3.0f%%:%.4f" % (epoch, int(rate * 100), loss), end="  ")
        losses.append(loss)

        net.eval()  # 测试
        train_acc = utils.eval_on_dataloader("train", train_loader, train_data.__len__(), net)
        train_accs.append(train_acc)

        test_acc = utils.eval_on_dataloader("test", test_loader, test_data.__len__(), net)
        test_accs.append(test_acc)

        print("train_acc:", train_acc, " test_acc:", test_acc)

        if train_acc==1.0 and test_acc==1.0:
            """保存模型"""
            torch.save({'state_dict': net.state_dict()}, pwd+'\model_save\model.pth.tar')
            print("Perfect!Done!")

    """可视化loss"""
    plot_loss_result(losses)
    plt.show()

    """训练集与测试集精度可视化"""
    plot_train_and_test_result(train_accs, test_accs)
    plt.show()


if __name__ == '__main__':
    main()
