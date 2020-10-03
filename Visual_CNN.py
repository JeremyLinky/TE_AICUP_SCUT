import torch
from torchviz import make_dot
import utils

"""该文件用于可视化神经网络结构"""
"""需分别在电脑和python环境安装graphviz，并手动添加graphviz/bin目录进入电脑的环境变量"""

models = utils.SimpleNet()
x = torch.randn(1, 12, 8, 8)
net_plot = make_dot(models(x),params = dict(models.named_parameters()))
net_plot.view()
