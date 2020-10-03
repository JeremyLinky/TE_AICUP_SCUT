import torch
import utils

def main():

    file_name = '0930-2_NOK_20200929114544.csv' # 数据文件名称

    """构造数据集"""
    feature = utils.process_data(file_name)
    feature = torch.tensor(feature, dtype=torch.float32)
    """加载模型"""
    model_load = utils.SimpleNet()
    checkpoint = torch.load('D:\大三下\AI CUP\Process\model_save\model.pth.tar') # 加载训练好的模型
    model_load.load_state_dict(checkpoint['state_dict'])
    outputs = model_load(feature.reshape(1, 12, 8, 8))

    predict = torch.max(outputs, dim=1)[1]
    if predict==0:
        print("铆压结果为：NOK")
    else:
        print("铆压结果为：OK")

if __name__ == '__main__':
    main()