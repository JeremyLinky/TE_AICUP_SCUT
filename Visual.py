import pandas as pd
import os
import Visual_utils

def main():

    namelist = os.listdir('D:/大三下/AI CUP/铆压机导出的数据') #数据目录

    for i in range(0,len(namelist)):

        Ndir = os.path.join('D:/大三下/AI CUP/铆压机导出的数据', namelist[i]);  # 保存可视化后的折线图
        csv_path = Ndir  # 此处要将“\”替换为“/”
        os.chdir(os.path.dirname(csv_path))  # 用os库改变目录

        data = pd.read_csv(os.path.basename(csv_path), header=11, usecols=[4, 6, 7, 9, 10], #取特定列的数据
                           error_bad_lines=False)  #用os.path.basename(csv_path)获取文件名

        Visual_utils.visualize(data,namelist[i]) #可视化压力、速度虽位移的变化曲线


if __name__ == '__main__':
    main()