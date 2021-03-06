import torch
import os
import utils
pwd = os.getcwd()  # 当前目录
def main():
    namelist = os.listdir(pwd+'/data')  # 数据目录

    """构造数据集"""
    feature, label = utils.prepare_data(namelist)
    feature = torch.tensor(feature, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    """加载模型"""
    model_load = utils.SimpleNet()
    checkpoint = torch.load(pwd+'\model_save\model.pth.tar') # 加载训练好的模型
    model_load.load_state_dict(checkpoint['state_dict'])
    """输出结果"""
    outputs = model_load(feature.reshape(len(namelist), 12, 8, 8))
    inference_acc = utils.eval_on_output(label, outputs)
    #print(label)
    #print(outputs)
    print("testing accuracy:",inference_acc)

if __name__ == '__main__':
    main()